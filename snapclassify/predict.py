"""Serving: one photo in, the top three breeds or "not sure" out, with NumPy and ONNX Runtime only.

This file must never import torch or torchvision: the app installs from app/requirements.txt, which has neither,
so it starts fast and fits the free host's memory.

    preprocess()   the evaluation transform, rewritten with PIL and NumPy                    TODO
    predict()      softmax, the top three, and "not sure" below the threshold                TODO

    python -m snapclassify.predict my_pet.jpg
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from snapclassify.constants import CROP, MEAN, RESIZE, STD

MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "model"


def load_model(model_dir: str | Path = MODEL_DIR):
    """(ONNX Runtime session, breed names, threshold) from app/model/. Given."""
    import onnxruntime as ort
    model_dir = Path(model_dir)
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return session, labels, float(config.get("threshold", 0.5))



def preprocess(image: Image.Image) -> np.ndarray:
    """Convert an image into an ONNX-ready normalized array."""

    image = image.convert("RGB")

    width, height = image.size

    if width < height:
        new_w = RESIZE
        new_h = int(RESIZE * height / width)
    else:
        new_h = RESIZE
        new_w = int(RESIZE * width / height)

    image = image.resize((new_w, new_h), Image.BILINEAR)

    left = int(round((new_w - CROP) / 2))
    top = int(round((new_h - CROP) / 2))

    image = image.crop((left, top, left + CROP, top + CROP))

    arr = np.asarray(image).astype(np.float32) / 255.0

    arr = (
        arr - np.array(MEAN, dtype=np.float32)
    ) / np.array(STD, dtype=np.float32)

    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)




def predict(image: Image.Image, session, labels: list[str], threshold: float = 0.5, k: int = 3) -> dict:
    """{"suggestion": the top breed, or None when not sure; "not_sure": bool;
        "top": [{"breed", "confidence"}, ...] (k of them, most confident first)}"""

    # 1. Get model predictions (logits)
    logits = session.run(None, {"image": preprocess(image)})[0][0]

    # 2. Convert logits into probabilities using stable softmax
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits)

    # 3. Get the indices of the top k predictions
    top_indices = np.argsort(-probs)[:k]

    # 4. Build the top k results
    top = [
        {
            "breed": labels[i],
            "confidence": float(probs[i])
        }
        for i in top_indices
    ]

    # 5. Check confidence threshold
    not_sure = bool(top[0]["confidence"] < threshold)

    # 6. Return the result
    suggestion = None if not_sure else top[0]["breed"]

    return {
        "suggestion": suggestion,
        "not_sure": not_sure,
        "top": top
    }



def main(argv: list[str] | None = None) -> int:
    paths = (argv if argv is not None else sys.argv[1:]) or []
    if not paths:
        print("usage: python -m snapclassify.predict PHOTO [PHOTO ...]")
        return 2
    session, labels, threshold = load_model()
    for p in paths:
        t0 = time.perf_counter()
        r = predict(Image.open(p), session, labels, threshold)
        ms = (time.perf_counter() - t0) * 1000
        head = "not sure" if r["not_sure"] else r["suggestion"]
        top = ", ".join(f"{t['breed']} {t['confidence']:.2f}" for t in r["top"])
        print(f"{p}: {head} ({top}) · {ms:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
