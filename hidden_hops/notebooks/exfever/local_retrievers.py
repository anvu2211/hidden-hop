"""Free, local retrievers over the EX-FEVER corpus, and a model-free comparison against gold.

Why this exists
---------------
Every benchmark in this project so far retrieves with OpenAI `text-embedding-3-large`. That is a
paid API, and it is the only reason indexing a corpus costs money at all -- the reader only ever
sees the top k, so corpus size hits the INDEX, not the reading. On a 5.2M-document corpus (HoVer)
that index is ~366M tokens, about $48. A local encoder removes the cost entirely and an RTX 3070
Ti runs one comfortably.

It also fixes a comparability problem. The published EX-FEVER / HoVer systems (GraphCheck,
ProgramFC, and the SCM-GRPO line that reuses GraphCheck's setting) retrieve with **BM25**, not a
dense encoder. Running BM25 here makes the `direct` arm comparable to their numbers in the
retrieval dimension as well as the prompt.

What was considered and rejected
--------------------------------
**GRITHopper-7B** (UKPLab, EACL 2026, `UKPLab/GritHopper-7B`) is the scientifically ideal choice:
state-of-the-art multi-hop dense retrieval, trained on EX-FEVER and HoVer among others. It is
built on GritLM-7B (Mistral-7B), so fp16 weights are ~14 GB against this machine's 8.6 GB of
VRAM, `bitsandbytes` and `accelerate` are not installed, and no smaller checkpoint exists. Revisit
on a >=24 GB GPU.

**MDR** (Xiong et al., ICLR 2021) is EX-FEVER's own retriever (their Table 3: EM 43.3, hit@6 55.0)
and is small enough, but the checkpoints and code are 2021-era with old dependencies. Worth trying
only if the two below are not enough.

Usage
-----
    python local_retrievers.py --build bm25          # seconds, CPU
    python local_retrievers.py --build bge           # ~10-20 min on the 3070 Ti, one time
    python local_retrievers.py --diagnose            # recall vs gold for every built index, $0
"""
import argparse, json, os, pickle, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "exfever_corpus.json")
CLAIMS = os.path.join(HERE, "exfever_graphcheck_100.json")
BGE_MODEL = "BAAI/bge-large-en-v1.5"
BGE_NPY = os.path.join(HERE, "emb_bge_large_en_v15.npy")
BM25_DIR = os.path.join(HERE, "bm25_index")


def load_corpus():
    with open(CORPUS, encoding="utf-8") as fh:
        raw = json.load(fh)
    out, seen = [], set()
    for p in raw:                       # same dedup rule as corpus_retriever.load_pooled
        t = (p.get("text") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append({"title": (p.get("title") or "").strip(), "text": t})
    return out


def render(p):
    return f"{p['title']}: {p['text']}".strip()


# --------------------------------------------------------------------------------- BM25 (bm25s)
class BM25Retriever:
    """Lexical BM25, the retrieval method GraphCheck and ProgramFC actually use on this dataset.

    Exposes the same two members the tree needs as CorpusRetriever: `.passages` and
    `.retrieve(query, k) -> [(passage, score)]`.
    """

    def __init__(self, passages, index_dir=BM25_DIR):
        import bm25s
        self.passages = list(passages)
        self.bm25 = bm25s.BM25.load(index_dir, mmap=False)
        self.tokenize = bm25s.tokenize

    def retrieve(self, query, k):
        import bm25s
        toks = bm25s.tokenize(str(query), show_progress=False)
        idx, sc = self.bm25.retrieve(toks, k=min(k, len(self.passages)), show_progress=False)
        return [(self.passages[int(i)], float(s)) for i, s in zip(idx[0], sc[0])]


def build_bm25(passages):
    import bm25s
    t0 = time.time()
    corpus = [render(p) for p in passages]
    tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(tokens, show_progress=False)
    bm25.save(BM25_DIR)
    print(f"bm25: indexed {len(corpus)} passages in {time.time()-t0:.0f}s -> {BM25_DIR}")


# ---------------------------------------------------------------------------------- BGE (dense)
class BGERetriever:
    """`BAAI/bge-large-en-v1.5` on the local GPU. Free; the index is a plain .npy like the OpenAI
    one, so nothing downstream changes.

    BGE was trained with an asymmetric convention: queries carry a retrieval instruction prefix
    and documents do not. Omitting it costs a few points of recall, so it is applied here.
    """
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, passages, npy=BGE_NPY, device=None):
        from sentence_transformers import SentenceTransformer
        import torch
        self.passages = list(passages)
        self.M = np.load(npy)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(BGE_MODEL, device=dev)
        self._qcache = {}

    def retrieve(self, query, k):
        q = str(query)
        v = self._qcache.get(q)
        if v is None:
            v = self.model.encode([self.QUERY_PREFIX + q], normalize_embeddings=True,
                                  show_progress_bar=False)[0].astype(np.float32)
            self._qcache[q] = v
        sims = self.M @ v
        k = min(k, len(self.passages))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.passages[i], float(sims[i])) for i in idx]


def build_bge(passages, batch=64):
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"bge: encoding {len(passages)} passages with {BGE_MODEL} on {dev} ...")
    model = SentenceTransformer(BGE_MODEL, device=dev)
    t0 = time.time()
    M = model.encode([render(p) for p in passages], batch_size=batch,
                     normalize_embeddings=True, show_progress_bar=True,
                     convert_to_numpy=True).astype(np.float32)
    np.save(BGE_NPY, M)
    print(f"bge: {M.shape} in {time.time()-t0:.0f}s -> {BGE_NPY}")


# ------------------------------------------------------------------------------- the diagnostic
def diagnose(ks=(1, 5, 10, 20)):
    """Recall of the gold documents for the claim as written. No reader calls, no API cost.

    Two quantities, as in run_exfever.recall_only: `any` gold in the top k (is the corpus hard)
    and `ALL` gold in the top k (can one retrieval finish the job -- the thing decomposition
    exists to fix).
    """
    passages = load_corpus()
    with open(CLAIMS, encoding="utf-8") as fh:
        claims = json.load(fh)
    title_of = {id(p): p["title"] for p in passages}

    retrievers = {}
    if os.path.exists(os.path.join(BM25_DIR, "data.csc.index.npy")) or os.path.isdir(BM25_DIR):
        try:
            retrievers["BM25 (GraphCheck's method)"] = BM25Retriever(passages)
        except Exception as e:
            print(f"  bm25 unavailable: {str(e)[:100]}")
    if os.path.exists(BGE_NPY):
        retrievers["BGE-large-en-v1.5 (local, free)"] = BGERetriever(passages)
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(HERE)), ".env"))
        sys.path.insert(0, os.path.dirname(HERE))
        from corpus_retriever import CorpusRetriever
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        client = (OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1") if key
                  else OpenAI())
        retrievers["text-embedding-3-large (paid)"] = CorpusRetriever(
            passages, client, HERE, verbose=False)
    except Exception as e:
        print(f"  openai index unavailable: {str(e)[:100]}")

    if not retrievers:
        sys.exit("nothing built yet; run --build bm25 and --build bge first")

    print(f"\nretrieval on the claim as written, {len(claims)} claims, {len(passages)} documents")
    print(f"{'retriever':<34} " + "  ".join(f"any@{k:<3} ALL@{k:<3}" for k in ks))
    kmax = max(ks)
    for name, r in retrievers.items():
        anyh = {k: 0 for k in ks}
        allh = {k: 0 for k in ks}
        for c in claims:
            got = [p["title"] for p, _ in r.retrieve(c["question"], kmax)]
            gold = set(c["gold_docs"])
            for k in ks:
                top = set(got[:k])
                anyh[k] += bool(gold & top)
                allh[k] += gold <= top
        n = len(claims)
        print(f"{name:<34} " + "  ".join(
            f"{100*anyh[k]/n:6.1f} {100*allh[k]/n:6.1f}" for k in ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", choices=["bm25", "bge"])
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()
    if args.build:
        passages = load_corpus()
        (build_bm25 if args.build == "bm25" else build_bge)(passages)
    if args.diagnose:
        diagnose()


if __name__ == "__main__":
    main()
