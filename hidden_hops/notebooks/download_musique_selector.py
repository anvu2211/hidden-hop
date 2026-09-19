"""Download and convert the official MuSiQue-Ans paragraph selector."""
from pathlib import Path
import hashlib
import gdown
from paper_selector import GDRIVE_ID_SELECTOR_ANS, convert_allennlp_archive

here = Path(__file__).resolve().parent
models = here / "models"
models.mkdir(exist_ok=True)
archive = models / "musique_sa_selector_ans_model.tar.gz"
output = models / "musique_sa_selector_ans.pt"
gdown.download(id=GDRIVE_ID_SELECTOR_ANS, output=str(archive), quiet=False)
convert_allennlp_archive(str(archive), str(output))
digest = hashlib.sha256(output.read_bytes()).hexdigest()
expected = "a0a4f05bba24d48c45b6adb2275a4a7ac54c1c02de045b357c48deab6e3303c1"
if digest != expected:
    raise SystemExit(f"checkpoint checksum mismatch: {digest}")
archive.unlink()
print(f"ready: {output}")
