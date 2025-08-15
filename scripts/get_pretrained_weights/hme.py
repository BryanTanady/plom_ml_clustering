from huggingface_hub import hf_hub_download
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
dest_path = ROOT / "weights" / "pretrained"

hf_hub_download(
    repo_id="bryantanady/plom_ml_HME",
    filename="hme_symbolic_model.pth",
    local_dir=dest_path.parent,
)

hf_hub_download(
    repo_id="bryantanady/plom_ml_HME",
    filename="hme_symbolic_model.onnx",
    local_dir=dest_path.parent,
)
