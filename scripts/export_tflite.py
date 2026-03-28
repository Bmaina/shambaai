#!/usr/bin/env python3
"""
scripts/export_tflite.py — Export CropDiseaseNet to TFLite for Android deployment.

Conversion pipeline:
    PyTorch (.pt) → ONNX (.onnx) → TF SavedModel → TFLite INT8 (.tflite)

INT8 quantisation:
  - Reduces model size ~4× (from ~11MB to ~3MB)
  - Speeds up inference ~2× on ARM CPUs
  - Acceptable accuracy drop: <1% on Top-1 accuracy

Usage:
    pip install onnx onnx-tf tensorflow
    python scripts/export_tflite.py --checkpoint models/disease_cnn_best.pt

Output:
    models/shambaai_disease.onnx
    models/shambaai_disease.tflite
    models/shambaai_disease_metadata.json  ← for Android integration
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import torch
from shambaai.models.disease_cnn import CropDiseaseNet, DISEASE_CLASSES, TREATMENT_MAP


def export_onnx(model: CropDiseaseNet, output_path: str):
    """Step 1: PyTorch → ONNX"""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model.backbone, dummy, output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}},
        opset_version=12,
        verbose=False,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  [ONNX] Exported → {output_path} ({size_mb:.1f} MB)")


def export_tflite_instructions(onnx_path: str, output_dir: str):
    """Step 2+3: Print TFLite conversion commands (requires tensorflow)."""
    saved_model_dir = os.path.join(output_dir, "saved_model")
    tflite_path     = os.path.join(output_dir, "shambaai_disease.tflite")

    print(f"\n  [TFLite] To complete conversion, run:")
    print(f"\n    # Install conversion tools (one-time):")
    print(f"    pip install onnx-tf tensorflow")
    print(f"\n    # Step 2: ONNX → TF SavedModel")
    print(f"    onnx-tf convert -i {onnx_path} -o {saved_model_dir}/")
    print(f"\n    # Step 3: TF SavedModel → TFLite INT8")
    print(f"    python3 - << 'EOF'")
    print(f"import tensorflow as tf")
    print(f"converter = tf.lite.TFLiteConverter.from_saved_model('{saved_model_dir}')")
    print(f"converter.optimizations = [tf.lite.Optimize.DEFAULT]")
    print(f"converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]")
    print(f"converter.inference_input_type = tf.int8")
    print(f"converter.inference_output_type = tf.int8")
    print(f"tflite_model = converter.convert()")
    print(f"open('{tflite_path}', 'wb').write(tflite_model)")
    print(f"print('Model size:', len(tflite_model)/1e6, 'MB')")
    print(f"EOF")


def export_metadata(output_dir: str, num_classes: int, class_names: list):
    """Generate metadata JSON for Android integration."""
    metadata = {
        "model_name": "ShambaAI CropDiseaseNet",
        "version": "1.0.0",
        "input": {
            "shape": [1, 3, 224, 224],
            "dtype": "float32",
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
            "description": "RGB crop photo, normalised to ImageNet stats"
        },
        "output": {
            "shape": [1, num_classes],
            "dtype": "float32",
            "description": "Logits — apply softmax for probabilities"
        },
        "classes": [
            {
                "index": i,
                "name": cls,
                "treatment": TREATMENT_MAP.get(cls, {})
            }
            for i, cls in enumerate(class_names)
        ],
        "performance": {
            "inference_target_ms": 3000,
            "min_android_api": 21,
            "min_ram_mb": 512,
        },
        "deployment": {
            "whatsapp_bot": "Farmer sends photo → receive prediction JSON → format response",
            "android_integration": "Use TFLite Interpreter API with GPU delegate",
            "ussd_fallback": "If photo diagnosis unavailable, use satellite score only"
        }
    }
    path = os.path.join(output_dir, "shambaai_disease_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  [Meta] Metadata → {path}")
    return metadata


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = CropDiseaseNet(num_classes=args.num_classes, pretrained=False)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            class_names = ckpt.get("class_names", DISEASE_CLASSES)
            num_classes  = ckpt.get("num_classes", args.num_classes)
        else:
            model.load_state_dict(ckpt)
            class_names = DISEASE_CLASSES
            num_classes  = args.num_classes
        print(f"  [Load] Checkpoint: {args.checkpoint}")
    else:
        print(f"  [Load] No checkpoint found — using random weights (demo)")
        class_names = DISEASE_CLASSES
        num_classes  = args.num_classes

    # Step 1: Export ONNX
    onnx_path = os.path.join(args.output_dir, "shambaai_disease.onnx")
    export_onnx(model, onnx_path)

    # Step 2+3: Print TFLite instructions
    export_tflite_instructions(onnx_path, args.output_dir)

    # Metadata
    export_metadata(args.output_dir, num_classes, class_names)

    print(f"\n  [Export] Complete. Files in {args.output_dir}/")
    print(f"    shambaai_disease.onnx")
    print(f"    shambaai_disease_metadata.json")
    print(f"    (run tflite conversion above to get .tflite)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ShambaAI to TFLite")
    parser.add_argument("--checkpoint",  default="models/disease_cnn_best.pt")
    parser.add_argument("--output_dir",  default="models/")
    parser.add_argument("--num_classes", type=int, default=40)
    args = parser.parse_args()
    main(args)
