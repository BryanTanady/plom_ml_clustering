import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from transformers import VisionEncoderDecoderModel
import torch
from pathlib import Path
import os

ROOT = Path(__file__).parent.parent.parent

FULL_ENCODER_OUTPUT = ROOT / "weights" / "trOCR_encoder.onnx"
HALF_PREC_ENCODER_OUTPUT = ROOT / "weights" / "trOCR_encoder_quantized.onnx"

# First, we download the trocr encoder in onnx
print("1. Downloading trOCR encoder in ONNX format")
model = VisionEncoderDecoderModel.from_pretrained(
    "fhswf/TrOCR_Math_handwritten"
)
encoder = model.encoder.eval()  # we only want encoder

# Dummy input (size must match model)
dummy_input = torch.randn(1, 3, 384, 384)

# Export encoder to ONNX
torch.onnx.export(
    encoder,
    dummy_input,
    FULL_ENCODER_OUTPUT,
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"}
    },
    opset_version=17
)
print("Downloaded encoder at " + str(FULL_ENCODER_OUTPUT))


print("2. Converting encoder to half precision ONNX")
# Then we convert that to half point precision
m = onnx.load(FULL_ENCODER_OUTPUT)

m16 = convert_float_to_float16(
    m, keep_io_types=True, disable_shape_infer=True
)

onnx.save_model(
    m16, 
    str(HALF_PREC_ENCODER_OUTPUT),
    save_as_external_data=False
)
print("Done, saved at: " + str(HALF_PREC_ENCODER_OUTPUT))
