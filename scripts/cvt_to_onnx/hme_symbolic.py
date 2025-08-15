import torch
import argparse
from model.model_architecture import HMESymbolicNet


def export(model_path, out_path):
    ckpt = torch.load(model_path)
    model = HMESymbolicNet(emb_dim=128)
    model.load_state_dict(ckpt)
    dummy_input = torch.randn(1, 1, 128, 256)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HME model pth to ONNX")
    parser.add_argument("model_path", help="path to HME_Symbolic pth")
    parser.add_argument(
        "--out", default="weights/hme_symbolic.onnx", help="Output ONNX file path"
    )

    args = parser.parse_args()
    export(args.model_path, args.out)
    print("Saved at: ", args.out)
