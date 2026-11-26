"""Shared preprocessing and CPU inference for Detective Rice."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def read_metadata(folder=ROOT):
    return json.loads((folder / "model_metadata.json").read_text(encoding="utf-8"))


def preprocess(photo, metadata):
    settings = metadata["preprocessing"]
    rgb = ImageOps.exif_transpose(photo).convert("RGB")
    view = ImageOps.pad(
        rgb, tuple(settings["size"]), Image.Resampling.BILINEAR,
        color=tuple(settings["padding_rgb"]), centering=tuple(settings["centering"]),
    )
    pixels = np.asarray(view, dtype=np.float32) / np.float32(255)
    mean = np.array(settings["mean"], dtype=np.float32)
    std = np.array(settings["std"], dtype=np.float32)
    tensor = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1))
    return tensor, view


def pytorch_model(class_count):
    from torch import nn
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
    return model


def load_predictor(backend, metadata, folder=ROOT):
    if backend == "ONNX Runtime":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            str(folder / "rice_variety_model.onnx"), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        embedded = session.get_modelmeta().custom_metadata_map.get("detective_rice")
        if embedded is None or json.loads(embedded) != metadata:
            raise ValueError("The ONNX file and model_metadata.json are from different exports.")

        def predict(batch):
            return session.run(["logits"], {"images": batch})[0]

        return predict

    if backend == "PyTorch":
        import torch

        torch.set_num_threads(2)
        saved = torch.load(folder / "rice_variety_model.pth", map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("The PyTorch file and model_metadata.json are from different exports.")
        model = pytorch_model(len(metadata["classes"]))
        model.load_state_dict(saved["state_dict"])
        model.eval()

        @torch.inference_mode()
        def predict(batch):
            return model(torch.from_numpy(batch)).numpy()

        return predict

    raise ValueError(f"Unknown backend: {backend}")


def scores(logits):
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()
