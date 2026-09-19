"""Build an EX-FEVER evaluation set in the schema run_exfever / eval_variant already speak.

Why EX-FEVER
------------
The supervisor asked what this method is *for*. The boundary condition established by
`bioasq/README.md` says decomposition pays exactly when the composed question cannot retrieve its
own evidence, and the adaptive half pays only when the question has hidden hops -- a bridge entity
that is never named, so no rewrite of the query can find it. Multi-hop fact verification is that
condition by construction: a real claim refers to its bridge entity by description ("the film
produced by an American comedian known as the founder of Apatow Productions"), never by name.

EX-FEVER is the one multi-hop fact-verification benchmark that is a drop-in here, for a reason
that is not a matter of taste: **it ships its own corpus**. `data/wiki_db.db` is a 50,231-document
SQLite index, and every gold entity of every split resolves against it (checked: 29,482/29,482 on
dev, 100%). HoVer, the other candidate, points at the 5.2M-document HotpotQA Wikipedia dump, which
is ~366M embedding tokens and a 64 GB float32 matrix in `corpus_retriever` -- not runnable here
without inventing a non-standard pooled corpus.

What is built
-------------
claims   `data/mini_test.csv`, the 1,000-claim split **the EX-FEVER authors themselves used for
         their LLM evaluation** (Ma et al., ACL Findings 2024, Table 6). Using their LLM split
         rather than the 12,059-row dev set is what makes our numbers sit in the same table as
         their ChatGPT rows. Balanced: 326 SUPPORT / 353 REFUTE / 321 NOT ENOUGH INFO, and
         541 two-hop / 459 three-hop by gold-document count.

corpus   All 50,231 documents in `wiki_db.db`, unchanged. 15,762 of them are train+dev gold, so
         roughly 69% are distractors: retrieval binds, which is the precondition the BioASQ port
         had to go looking for. One passage = one Wikipedia article, the same unit EX-FEVER's own
         document-retrieval baselines (MDR, BERT-based) are scored on.

Hidden-hop diagnostic (model-free, printed at build time)
---------------------------------------------------------
For each claim, how many of its gold documents are named verbatim in the claim text? This is the
zero-API-call predictor that called both the MuSiQue (+25 F1) and BioASQ (null) outcomes before a
single reader call was spent. Verbatim matching under-counts paraphrase, so the "named" figure is
an upper bound and the hidden-hop share is a lower bound.

    python build_exfever.py            # -> exfever_claims.json + exfever_corpus.json
"""
import argparse, ast, collections, csv, json, os, random, re, sqlite3, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
BASE = "https://raw.githubusercontent.com/dependentsign/EX-FEVER/master/data"
FILES = {"mini_test.csv": f"{BASE}/mini_test.csv",
         "dev.csv": f"{BASE}/dev.csv",
         "wiki_db.db": f"{BASE}/wiki_db.db"}
LABELS = ("SUPPORT", "REFUTE", "NOT ENOUGH INFO")


def fetch(name):
    """Download once into raw/. The repo's default branch is `master`, not `main`."""
    dest = os.path.join(RAW, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    os.makedirs(RAW, exist_ok=True)
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(FILES[name], dest)
    print(f"  -> {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
    return dest


def load_corpus(db_path):
    """[{title, text}] from wiki_db.db. `id` is the Wikipedia title with underscores."""
    con = sqlite3.connect(db_path)
    out = []
    for doc_id, text in con.execute("select id, text from documents"):
        t = " ".join((text or "").split())
        if not t:
            continue                      # 42 rows have empty text
        out.append({"title": (doc_id or "").replace("_", " ").strip(), "text": t})
    con.close()
    return out


def _norm_title(s):
    """'Judd_Apatow' / 'The Shining (film)' -> a bag of words for verbatim matching."""
    s = re.sub(r"\s*\([^)]*\)\s*$", "", str(s).replace("_", " "))
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def named_in(title, claim):
    t = _norm_title(title)
    c = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", claim.lower())) + " "
    return bool(t) and (" " + t + " ") in c


def build_claims(rows, split):
    out = []
    for i, r in enumerate(rows):
        label = (r.get("label") or "").strip().upper()
        if label not in LABELS:
            continue
        try:
            gold = [g for g in ast.literal_eval(r["golden entity"]) if g]
        except Exception:
            gold = []
        claim = " ".join((r.get("claim") or "").split())
        if not claim or not gold:
            continue
        out.append({
            "id": f"exfever_{split}_{i}",
            # `question` is the field the tree reads. The claim goes in verbatim: the point of a
            # transfer study is that the method does not get rewritten per benchmark.
            "question": claim,
            "claim": claim,
            "label": label,
            "gold_docs": [g.replace("_", " ").strip() for g in gold],
            "n_hops": len(gold),
            "explanation": " ".join((r.get("explanation") or "").split()),
        })
    return out


def diagnostics(claims):
    print("\nhidden-hop diagnostic (verbatim gold-title match in the claim; "
          "'named' is an UPPER bound)")
    print(f"  {'hops':>4} {'n':>5} {'mean frac named':>16} {'all named':>10} {'none named':>11}")
    for h in sorted({c["n_hops"] for c in claims}):
        sub = [c for c in claims if c["n_hops"] == h]
        fr = [sum(named_in(t, c["claim"]) for t in c["gold_docs"]) / h for c in sub]
        print(f"  {h:>4} {len(sub):>5} {sum(fr)/len(fr):>16.2f} "
              f"{100*sum(1 for v in fr if v == 1)/len(sub):>9.1f}% "
              f"{100*sum(1 for v in fr if v == 0)/len(sub):>10.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini_test", choices=["mini_test", "dev"])
    ap.add_argument("--labels", choices=["3way", "2way"], default="3way",
                    help="'2way' drops NOT ENOUGH INFO, which is what GraphCheck / SCM-GRPO / "
                         "ReflectFact do ('We exclude NEI-labeled samples'). It raises the "
                         "guessing floor from 33%% to 50%% and removes the hardest class, so 2way "
                         "and 3way numbers must never share a table.")
    ap.add_argument("--n", type=int, default=100,
                    help="sample size to write (0 = the whole split)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    db = fetch("wiki_db.db")
    src = fetch(f"{args.split}.csv")

    corpus = load_corpus(db)
    chars = [len(p["text"]) for p in corpus]
    print(f"corpus: {len(corpus)} documents, {sum(chars)/len(chars):.0f} chars mean, "
          f"{sum(chars)/1e6:.1f}M chars total")

    with open(src, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pool = build_claims(rows, args.split)
    print(f"{args.split}: {len(pool)} claims with a label and gold entities")
    if args.labels == "2way":
        before = len(pool)
        pool = [c for c in pool if c["label"] != "NOT ENOUGH INFO"]
        print(f"  --labels 2way: dropped {before-len(pool)} NOT ENOUGH INFO claims, "
              f"{len(pool)} left")
    print(f"  labels {dict(collections.Counter(c['label'] for c in pool))}")
    print(f"  hops   {dict(sorted(collections.Counter(c['n_hops'] for c in pool).items()))}")

    titles = {p["title"] for p in corpus}
    missing = sum(1 for c in pool for g in c["gold_docs"] if g not in titles)
    total = sum(len(c["gold_docs"]) for c in pool)
    print(f"  gold documents present in the corpus: {total-missing}/{total} "
          f"({100*(total-missing)/total:.1f}%)")
    if missing:
        sys.exit("gold documents are missing from the corpus; refusing to build a set whose "
                 "questions are unanswerable by construction")

    # One fixed sample per seed, drawn from a sorted pool so the draw does not depend on the
    # file's own ordering -- the discipline eval_variant adopted after the unpaired-sampling bug.
    pool.sort(key=lambda e: e["id"])
    if args.n and args.n < len(pool):
        rng = random.Random(args.seed)
        pool = rng.sample(pool, args.n)
        pool.sort(key=lambda e: e["id"])
        print(f"sampled {len(pool)} with seed {args.seed}: "
              f"labels {dict(collections.Counter(c['label'] for c in pool))}, "
              f"hops {dict(sorted(collections.Counter(c['n_hops'] for c in pool).items()))}")

    prefix = args.out_prefix or (f"exfever_{args.split}"
                                 f"{'_2way' if args.labels == '2way' else ''}"
                                 f"{('_' + str(args.n)) if args.n else ''}")
    qf = os.path.join(HERE, f"{prefix}.json")
    cf = os.path.join(HERE, "exfever_corpus.json")
    with open(qf, "w", encoding="utf-8") as fh:
        json.dump(pool, fh, ensure_ascii=False)
    print(f"wrote {qf}")
    if not os.path.exists(cf):
        with open(cf, "w", encoding="utf-8") as fh:
            json.dump(corpus, fh, ensure_ascii=False)
        print(f"wrote {cf}")
    else:
        # The embedding cache is keyed on the corpus text, so silently rewriting this file would
        # invalidate a 50k-document index build. Refuse and say so. (Same guard as BioASQ.)
        print(f"kept existing {cf} (delete it to rebuild; the embedding cache is keyed on it)")

    diagnostics(pool)


if __name__ == "__main__":
    main()
