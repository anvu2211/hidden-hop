# Hidden Hops: code and result logs

This is the supplementary code-and-results package for the paper. It contains the three experiment
notebooks, the lightweight local modules they import, the manual FC-MH annotation, and every result
log used by the current tables and appendices. Superseded prompts, deleted notebook drafts, editor
checkpoints, and experiments no longer cited by the paper are excluded.

## Layout

```text
notebooks/
  01_fcmh_runs.ipynb
  02_musique_runs.ipynb
  03_exfever_runs.ipynb
  paper_selector.py
  download_musique_selector.py
  exfever/{build_exfever.py,local_retrievers.py,README.md}
results/
  fcmh/                 FC-MH with gpt-4o-mini, including controls and depth sweep
  fcmh_other_readers/   FC-MH with Luna, Qwen, and Gemma
  musique/              MuSiQue-Ans, full development split
  exfever/              two EX-FEVER 1,000-claim runs
  annotation/           manual FC-MH hidden-hop annotation
requirements.txt
```

## Setup

Python 3.11 was used. From the package root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name hidden-hops
cd notebooks
```

Set `OPENAI_API_KEY` for direct OpenAI calls. FC-MH's optional OpenRouter reader cells and the
MuSiQue notebook use `OPENROUTER_API_KEY`. API keys are not included.

### Dataset and retriever assets

- **FC-MH:** notebook 1 downloads `ai-hyz/MemoryAgentBench`, split `Conflict_Resolution`, through
  Hugging Face Datasets and selects `factconsolidation_mh_262k`.
- **MuSiQue:** notebook 2 downloads the MuSiQue-Ans development split through Hugging Face. Before
  running it, execute `python download_musique_selector.py`. 
- **EX-FEVER:** notebook 3 uses the included builder to download the authors' mini-test and
  Wikipedia database. It then builds BM25 and `BAAI/bge-large-en-v1.5` indexes locally.

Run Jupyter from `notebooks/` or from the package root.

## Result provenance

| Paper result | Released files |
|---|---|
| FC-MH main table and depth sweep | `results/fcmh/` |
| FC-MH reader comparison | `results/fcmh_other_readers/` plus `results/fcmh/` |
| MuSiQue main table, depth sweep, traces, and variance | `results/musique/` |
| EX-FEVER main table and retrieval analysis | `results/exfever/` |

File names use `<dataset>_[reader_]<procedure>_k<depth>_run<n>.json`. `run<n>` denotes an
independent execution. `_traced`, `_repaired`, and `_sharedplan` identify the logged trace, repaired
provider-error rows, and reused plan respectively.

Scores with multiple runs are reported as mean ± sample standard deviation. FC-MH paired intervals
and exact McNemar tests operate on per-question correctness. MuSiQue intervals and sign-flip tests
operate on run-averaged per-question F1. Trigger/Split % is the percentage of questions with at
least one executed split; it is different from the refusal rate.
