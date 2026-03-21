"""
backend/models/local_model.py
──────────────────────────────
Wrapper for YOUR trained model weights (Keras .h5 or ONNX .onnx).

HOW TO PLUG IN YOUR MODEL
──────────────────────────
1. Set USE_LOCAL_MODEL=true in .env
2. Set MODEL_PATH=path/to/your/weights in .env
3. Set MODEL_TYPE=keras  (for .h5)  or  MODEL_TYPE=onnx  (for .onnx)
4. Adjust _preprocess() and _postprocess() to match what your model expects.

Keras models
────────────
Default assumption: model takes a (1, H, W, 1) float32 array normalised
to [0, 1] and returns a (1, H, W, 1) float32 mask where > 0.5 = ink.
Change INPUT_SIZE to match your model's expected resolution.

ONNX models
───────────
Default assumption: single input named "input" with shape (1, 1, H, W).
Adjust _run_onnx() if your model has a different layout.
"""

import logging
import numpy as np
from PIL import Image

from backend.models.base import BinarizationModel
from config.settings import MODEL_PATH, MODEL_TYPE

logger = logging.getLogger("AMDES.LocalModel")

# ── Change these to match your model ──────────────────────────────────────────
INPUT_SIZE   = (256, 256)   # (width, height) fed to the model
INK_THRESHOLD = 0.5         # sigmoid output above this = ink pixel
# ──────────────────────────────────────────────────────────────────────────────


class LocalModel(BinarizationModel):
    """Wraps a user-supplied Keras (.h5) or ONNX (.onnx) model."""

    def __init__(self):
        self._keras_model = None
        self._onnx_session = None
        self._load()

    def _load(self):
        if MODEL_TYPE == "keras":
            self._load_keras()
        elif MODEL_TYPE == "onnx":
            self._load_onnx()
        else:
            raise ValueError(f"Unknown MODEL_TYPE '{MODEL_TYPE}'. Use 'keras' or 'onnx'.")

    def _load_keras(self):
        try:
            import tensorflow as tf  # noqa: F401  (not in default requirements)
            self._keras_model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"✅ Keras model loaded from {MODEL_PATH}")
        except ImportError:
            raise RuntimeError(
                "TensorFlow is not installed. "
                "Add tensorflow to requirements.txt or switch to MODEL_TYPE=onnx."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model from {MODEL_PATH}: {e}")

    def _load_onnx(self):
        try:
            import onnxruntime as ort  # noqa: F401  (not in default requirements)
            self._onnx_session = ort.InferenceSession(
                MODEL_PATH, providers=["CPUExecutionProvider"]
            )
            logger.info(f"✅ ONNX model loaded from {MODEL_PATH}")
        except ImportError:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Add onnxruntime to requirements.txt."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {MODEL_PATH}: {e}")

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        """
        Convert PIL image → model input tensor.

        Returns a (1, H, W, 1) float32 array for Keras,
        or (1, 1, H, W) float32 for ONNX.
        Adjust here if your model expects RGB or a different shape.
        """
        gray = img.convert("L").resize(INPUT_SIZE)
        arr  = np.array(gray, dtype=np.float32) / 255.0  # normalise to [0, 1]

        if MODEL_TYPE == "keras":
            return arr[np.newaxis, ..., np.newaxis]        # (1, H, W, 1)
        else:  # onnx
            return arr[np.newaxis, np.newaxis, ...]         # (1, 1, H, W)

    # ── Postprocessing ────────────────────────────────────────────────────────

    def _postprocess(self, raw_output: np.ndarray, original_size) -> Image.Image:
        """
        Convert raw model output → binarized PIL Image.

        Adjust here if your model outputs logits, multi-class probabilities,
        or a different channel ordering.
        """
        # Squeeze batch + channel dims → 2-D (H, W)
        mask = raw_output.squeeze()
        binary = ((mask > INK_THRESHOLD) * 255).astype(np.uint8)
        result = Image.fromarray(binary, mode="L")
        return result.resize(original_size, Image.NEAREST)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_keras(self, tensor: np.ndarray) -> np.ndarray:
        return self._keras_model.predict(tensor, verbose=0)

    def _run_onnx(self, tensor: np.ndarray) -> np.ndarray:
        input_name  = self._onnx_session.get_inputs()[0].name
        output_name = self._onnx_session.get_outputs()[0].name
        return self._onnx_session.run([output_name], {input_name: tensor})[0]

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, img: Image.Image) -> Image.Image:
        original_size = img.size
        tensor        = self._preprocess(img)

        if MODEL_TYPE == "keras":
            raw = self._run_keras(tensor)
        else:
            raw = self._run_onnx(tensor)

        return self._postprocess(raw, original_size)

    @property
    def name(self) -> str:
        return f"LocalModel({MODEL_TYPE}, {MODEL_PATH})"
