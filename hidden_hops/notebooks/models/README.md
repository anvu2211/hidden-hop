# MuSiQue selector checkpoint

`02_musique_runs.ipynb` uses `musique_sa_selector_ans.pt`, a plain-PyTorch conversion of the
official MuSiQue Select+Answer paragraph selector. It is not included because the converted file is
1.42 GB and originates from the MuSiQue authors.

From the `notebooks/` directory, run:

```bash
python download_musique_selector.py
```

The script downloads the authors' archive from Google Drive id
`115dHg4q1TBbbVLL1zSrWt3LfPedsSxHR`, converts it with `paper_selector.py`, and writes
`models/musique_sa_selector_ans.pt`.

Expected SHA-256 of the converted checkpoint:
`a0a4f05bba24d48c45b6adb2275a4a7ac54c1c02de045b357c48deab6e3303c1`.
