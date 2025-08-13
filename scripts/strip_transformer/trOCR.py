from transformers import VisionEncoderDecoderModel, BitsAndBytesConfig
import torch

# Load in 8-bit
bnb = BitsAndBytesConfig(load_in_8bit=True)
model = VisionEncoderDecoderModel.from_pretrained(
    "fhswf/TrOCR_Math_handwritten",
    device_map="auto",
    quantization_config=bnb,
)

# Grab just the encoder
encoder = model.encoder.eval()

# Torch-pickle the entire module
torch.save(encoder, "trocr_math_encoder_8bit.pth")
