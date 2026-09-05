import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import numpy as np
from sat_seg_model import SatSegNet

def export_satsegnet_to_onnx():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_satsegnet.pth")
    output_onnx_path = os.path.join(base_dir, "checkpoints", "satsegnet.onnx")
    output_quant_path = os.path.join(base_dir, "checkpoints", "satsegnet_quantized.onnx")
    
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at {checkpoint_path}")
        return False

    print(f"[Export] Loading SatSegNet weights from {checkpoint_path}...")
    model = SatSegNet(n_channels=3, n_classes=6)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input: (batch_size=1, channels=3, height=128, width=128)
    dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float32)

    print(f"[Export] Exporting to ONNX format at {output_onnx_path}...")
    # Set dynamo=False to use the battle-tested, classic TorchScript ONNX exporter
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "logits": {0: "batch_size", 2: "height", 3: "width"}
        },
        dynamo=False
    )

    onnx_size = os.path.getsize(output_onnx_path) / 1024
    print(f"[Success] Exported ONNX model ({onnx_size:.1f} KB) successfully!")

    # Verify ONNX model with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)

        # PyTorch reference
        with torch.no_grad():
            torch_outs = model(dummy_input).numpy()

        diff = float(np.max(np.abs(ort_outs[0] - torch_outs)))
        print(f"[Verification] Max absolute difference between PyTorch & ONNX: {diff:.6e}")
        if diff < 1e-4:
            print("[Verification] PASSED: ONNX model produces identical outputs to PyTorch!")
        else:
            print(f"[Verification] Difference is {diff:.6e}")

        # Dynamic INT8 Quantization for ultra-low memory & fast CPU speed
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"[Quantization] Applying dynamic 8-bit quantization...")
            quantize_dynamic(
                model_input=output_onnx_path,
                model_output=output_quant_path,
                weight_type=QuantType.QUInt8
            )
            q_size = os.path.getsize(output_quant_path) / 1024
            print(f"[Success] Quantized ONNX model saved to {output_quant_path} ({q_size:.1f} KB)!")
        except Exception as q_err:
            print(f"[Quantization Notice] {q_err}")

    except Exception as ie:
        print(f"[Notice] Verification/Quantization: {ie}")

    return True

if __name__ == "__main__":
    export_satsegnet_to_onnx()
