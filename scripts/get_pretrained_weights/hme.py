from huggingface_hub import hf_hub_download
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
dest_path = ROOT / "weights" / "hme_symbolic_model_pretrained.pth"
temp_dest = dest_path.parent / "hme_symbolic_model.pth"
# Only rename if file doesn't already exist
if temp_dest.exists():
    raise FileExistsError(f"{temp_dest} already exists — refusing to overwrite.")

hf_hub_download(
    repo_id="bryantanady/plom_ml_HME",
    filename="mcq_model.pth",
    local_dir=dest_path.parent,
)

# Rename from mcq_model.pth to mcq_model_pretrained.pth
temp_dest.rename(dest_path)
