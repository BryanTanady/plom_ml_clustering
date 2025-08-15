from transformers import VisionEncoderDecoderModel, BitsAndBytesConfig
import torch

bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
model = VisionEncoderDecoderModel.from_pretrained(
    "fhswf/TrOCR_Math_handwritten",
    device_map="auto",
    quantization_config=bnb_cfg
)
encoder = model.encoder

torch.save(encoder.state_dict(), "weights/trOCR_encoder_quantized.pth")
