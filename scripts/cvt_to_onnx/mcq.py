import torch
import argparse
from model.model_architecture import MCQClusteringNet1


def export(model_path, out_path):
    ckpt = torch.load(model_path)
    model = MCQClusteringNet1(out_classes=11)
    model.load_state_dict(ckpt)
    dummy_input = torch.randn(1, 1, 64, 64)
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
    parser = argparse.ArgumentParser(description="Export model pth to ONNX")
    parser.add_argument("model_path", help="path to MCQ model pth")
    parser.add_argument(
        "--out", default="weights/mcq_model.onnx", help="Output ONNX file path"
    )

    args = parser.parse_args()
    export(args.model_path, args.out)
    print("Saved at: ", args.out)
