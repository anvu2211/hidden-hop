"""
Faithful port of the MuSiQue paper's Select+Answer *selector* (the retriever), out of AllenNLP.

Source: StonyBrookNLP/musique
  - model:  allennlp_lib/models/text_ranker.py            (TextRanker)
  - reader: allennlp_lib/data/dataset_readers/text_ranker.py (TextRankerReader)
  - config: experiment_configs/select_and_answer_model_selector_for_musique_ans.jsonnet

What the original does, verbatim:
  * roberta-large cross-encoder, scores each (question, paragraph) pair independently
  * input string = prefix_text + " SEP " + paragraph_text, where add_question_info=True
    prepends the question to prefix_text; raw contexts carry no prefix_text of their own,
    so in practice this is  "<question> SEP <paragraph_text>"
  * whitespace truncation to max_tokens=300 words, then subword truncation to 300 + specials
  * CLS (<s>, index 0) representation -> Linear(1024, 1) -> logit -> sigmoid
  * loss BCEWithLogits over every paragraph of the instance (has_single_positive=false)

This module reimplements exactly that in plain torch + transformers, with no AllenNLP.
"""

from __future__ import annotations

import io
import os
import tarfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Google Drive id of the authors' trained MuSiQue-Ans selector, taken from
# .all_experiment_information.json in the official repo.
GDRIVE_ID_SELECTOR_ANS = "115dHg4q1TBbbVLL1zSrWt3LfPedsSxHR"

MAX_TOKENS = 300           # jsonnet: dataset_reader.max_tokens
PREFIX_SEPARATOR = " SEP "  # reader: self._prefix_separator
BASE_MODEL = "roberta-large"


# --------------------------------------------------------------------------------------
# Input construction (TextRankerReader.json_to_instance)
# --------------------------------------------------------------------------------------
def build_input_text(question: str, paragraph_text: str, title: str = "",
                     use_title: bool = False) -> str:
    """Reproduce the reader's per-paragraph input string.

    use_title=False is the faithful setting (raw contexts have no prefix_text).
    use_title=True is the ablation: title folded in the way the *answerer* reader does it.
    """
    question = " ".join(question.split())          # reader's clean_ws
    prefix = question
    body = paragraph_text.strip()
    if use_title and title:
        body = f"{title.strip()} || {body}"
    text = PREFIX_SEPARATOR.join([prefix.strip(), body])
    return " ".join(text.split(" ")[:MAX_TOKENS])  # reader's word-level truncation


# --------------------------------------------------------------------------------------
# Model (TextRanker with seq2vec/feedforward/dropout all None, as configured)
# --------------------------------------------------------------------------------------
class TextRanker(nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, encoder=None, pretrained: bool = True):
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
        elif pretrained:
            self.encoder = AutoModel.from_pretrained(base_model)
        else:
            # random init: every weight gets overwritten by the checkpoint anyway, so this
            # skips downloading roberta-large and skips the pooler/lm_head mismatch warning
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(base_model))
        self.ranker = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids: (n_paragraphs, seq_len) -> logits: (n_paragraphs,)"""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden[:, 0]                       # _find_cls_index is always 0 after add_special_tokens
        return self.ranker(cls).squeeze(-1)


def load_selector(checkpoint_path: str, device: str = "cuda",
                  dtype: torch.dtype = torch.float16) -> TextRanker:
    """Load the converted checkpoint. Raises if any weight is missing or unexpected."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, base = blob["state_dict"], blob.get("base_model", BASE_MODEL)
    model = TextRanker(base, pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("embeddings.position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch. missing={missing} unexpected={unexpected}")
    return model.to(device=device, dtype=dtype).eval()


def convert_allennlp_archive(archive_path: str, out_path: str,
                             source: Optional[str] = None) -> str:
    """Strip AllenNLP's key prefixes out of model.tar.gz -> plain {encoder.*, ranker.*}.

    `source` records where the archive came from; pass it whenever you convert something other
    than the default MuSiQue-Ans SA selector, so the checkpoint does not misreport its origin.
    """
    enc_p = "_text_field_embedder.token_embedder_tokens.transformer_model."
    head_p = "_ranker_layer._module."
    with tarfile.open(archive_path) as tar:
        sd = torch.load(io.BytesIO(tar.extractfile("weights.th").read()), map_location="cpu")
    new = OrderedDict()
    for k, v in sd.items():
        if k.startswith(enc_p):
            nk = "encoder." + k[len(enc_p):]
            if nk.endswith("embeddings.position_ids"):
                continue                          # buffer, dropped by modern transformers
            new[nk] = v
        elif k.startswith(head_p):
            new["ranker." + k[len(head_p):]] = v
        else:
            raise KeyError(f"unmapped key {k}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if source is None:
        source = ("StonyBrookNLP/musique select_and_answer_model_selector_for_musique_ans "
                  f"(gdrive {GDRIVE_ID_SELECTOR_ANS})")
    torch.save({"state_dict": new, "base_model": BASE_MODEL, "source": source}, out_path)
    return out_path


# --------------------------------------------------------------------------------------
# Drop-in retriever with the same interface as the notebook's BM25 Retriever
# --------------------------------------------------------------------------------------
class SelectorRetriever:
    """Same API as the BM25 Retriever: .retrieve(query, k) -> [(passage, score), ...].

    Scores are the selector's sigmoid probabilities, so they are directly comparable across
    paragraphs and thresholdable at 0.5 the way the original TextRanker does for select_*.
    """

    def __init__(self, passages: Sequence[Dict], model: TextRanker, tokenizer,
                 device: str = "cuda", use_title: bool = False, batch_size: int = 32,
                 cache: bool = True):
        self.passages = list(passages)
        self.model, self.tok, self.device = model, tokenizer, device
        self.use_title, self.batch_size = use_title, batch_size
        self._cache: Dict[str, torch.Tensor] = {} if cache else None

    @torch.no_grad()
    def probs(self, query: str) -> torch.Tensor:
        if self._cache is not None and query in self._cache:
            return self._cache[query]
        texts = [build_input_text(query, p["text"], p.get("title", ""), self.use_title)
                 for p in self.passages]
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], add_special_tokens=True,
                           truncation=True, max_length=MAX_TOKENS + 2,
                           padding=True, return_tensors="pt").to(self.device)
            logits = self.model(enc["input_ids"], enc["attention_mask"]).float()
            out.append(logits.sigmoid().cpu())
        p = torch.cat(out) if out else torch.zeros(0)
        if self._cache is not None:
            self._cache[query] = p
        return p

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        p = self.probs(query)
        order = torch.argsort(p, descending=True)[:k].tolist()
        return [(self.passages[i], float(p[i])) for i in order]

    def select(self, query: str, threshold: float = 0.5) -> List[Tuple[Dict, float]]:
        """The paper's thresholded selection (no fixed k)."""
        p = self.probs(query)
        idx = (p > threshold).nonzero(as_tuple=True)[0].tolist()
        idx.sort(key=lambda i: -float(p[i]))
        return [(self.passages[i], float(p[i])) for i in idx]


def make_builder(checkpoint_path: str, device: str = "cuda", use_title: bool = False,
                 dtype: torch.dtype = torch.float16):
    """Returns (build_retriever, model, tokenizer) so a notebook can swap the backbone in
    one line while keeping a single model resident on the GPU."""
    model = load_selector(checkpoint_path, device=device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def build_retriever(inst):
        return SelectorRetriever(inst["paragraphs"], model, tok, device=device,
                                 use_title=use_title)

    return build_retriever, model, tok


# --------------------------------------------------------------------------------------
# The paper's own ranking metrics (allennlp_lib/training/metrics/)
# --------------------------------------------------------------------------------------
def list_compare_em_f1(pred_idx: Sequence[int], gold_idx: Sequence[int]) -> Tuple[float, float]:
    """ListCompareEmAndF1, the HotpotQA-style supporting-paragraph metric."""
    pred, gold = set(map(int, pred_idx)), set(map(int, gold_idx))
    if not pred and not gold:
        return 1.0, 1.0
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    em = 1.0 if pred == gold else 0.0
    return em, f1


def full_recall_rank(ordered_idx: Sequence[int], gold_idx: Sequence[int]) -> int:
    """FullRecallRank: how deep you must read the ranked list to have seen every gold
    paragraph. Lower is better; the answerer's take_predicted_topk_contexts is 7."""
    ordered = list(map(int, ordered_idx))
    gold = set(map(int, gold_idx))
    if not gold:
        return 100
    return max(ordered.index(g) for g in gold) + 1
