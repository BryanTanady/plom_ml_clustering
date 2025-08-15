from huggingface_hub import hf_hub_download
from pathlib import Path
import os

ROOT = Path(__file__).parent.parent.parent
dest_path = ROOT / "weights" / "pretrained"
os.makedirs(dest_path.parent, exist_ok=True)

hf_hub_download(
    repo_id="bryantanady/plom_ml_MCQ",
    filename="mcq_model.pth",
    local_dir=dest_path,
)

hf_hub_download(
    repo_id="bryantanady/plom_ml_MCQ",
    filename="mcq_model.onnx",
    local_dir=dest_path,
)
