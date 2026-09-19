# EX-FEVER generated assets

`03_exfever_runs.ipynb` calls `build_exfever.py` when the derived claim file is absent. The builder
downloads EX-FEVER's released `mini_test.csv` and `wiki_db.db`, then writes
`exfever_mini_test_all.json` and `exfever_corpus.json` here. The notebook then builds the BM25 and
BGE indexes when absent. These generated files are intentionally excluded from the submission:

- `raw/`
- `exfever_mini_test_all.json`
- `exfever_corpus.json`
- `bm25_index/`
- `emb_bge_large_en_v15.npy`

Building the BGE index downloads `BAAI/bge-large-en-v1.5`; a CUDA GPU is recommended.
