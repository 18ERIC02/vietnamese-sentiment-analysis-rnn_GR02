"""Module for loading and running inference with the trained Vietnamese RNN model.

Technical Note:
    The trained model uses Vanilla RNN architecture implemented via
    `tf.keras.layers.SimpleRNN`. In the user interface, it is displayed
    simply as 'RNN' according to design specifications.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import tensorflow as tf

# Standard label mapping for UIT-VSFC: 0: Negative, 1: Neutral, 2: Positive
LABEL_NAMES_VI = ["Tiêu cực", "Trung tính", "Tích cực"]
LABEL_NAMES_EN = ["Negative", "Neutral", "Positive"]


class ModelLoadError(Exception):
    """Raised when the trained model cannot be loaded."""
    pass


class PredictionError(Exception):
    """Raised when an error occurs during model inference."""
    pass


def get_app_dir() -> Path:
    """Return the base application directory whether running from source or frozen binary."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def find_checkpoint_path(base_dir: Path | None = None) -> Path:
    """Find the checkpoint path, handling frozen executables and possible trailing spaces."""
    app_dir = get_app_dir()
    search_dirs: list[Path] = []
    if base_dir:
        search_dirs.append(Path(base_dir))

    search_dirs.extend([
        app_dir / "models",
        app_dir / "_internal" / "models",
        Path(sys.executable).resolve().parent / "models" if getattr(sys, "frozen", False) else app_dir / "models",
        Path(__file__).resolve().parent.parent / "models",
        Path(__file__).resolve().parent.parent.parent / "rnn_demo" / "models",
        Path(__file__).resolve().parent.parent.parent / "Result",
    ])

    for d in search_dirs:
        for name in ["best_model.keras", "best_model .keras"]:
            candidate = d / name
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError(
        f"Không tìm thấy file checkpoint 'best_model.keras'. Đã tìm trong: {[str(d) for d in search_dirs]}"
    )


class SentimentPredictor:
    """Manages model loading and prediction for Vietnamese sentiment analysis."""

    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else find_checkpoint_path()
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._is_loaded = False

    def load(self) -> None:
        """Load the Keras model with UTF-8 patch for vocabulary on Windows."""
        if not self.model_path.is_file():
            raise ModelLoadError(f"File mô hình không tồn tại: {self.model_path}")

        # On Windows, Keras 3.x opens text assets using the default locale (cp1252),
        # which fails on Vietnamese utf-8 characters in vocabulary.txt.
        original_open = open

        def open_vocabulary_as_utf8(file, mode="r", *args, **kwargs):
            is_vocab = (
                isinstance(file, (str, bytes, Path))
                and Path(file).name == "vocabulary.txt"
            )
            if is_vocab and "b" not in mode and "encoding" not in kwargs:
                kwargs["encoding"] = "utf-8"
            return original_open(file, mode, *args, **kwargs)

        try:
            with patch("builtins.open", open_vocabulary_as_utf8):
                self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self._extract_metadata()
            self._is_loaded = True
        except Exception as e:
            self._is_loaded = False
            raise ModelLoadError(f"Không thể nạp mô hình từ {self.model_path}: {e}") from e

    def _extract_metadata(self) -> None:
        """Extract layer parameters and TextVectorization config."""
        if self.model is None:
            return

        by_type = {layer.__class__.__name__: layer for layer in self.model.layers}
        vectorizer = by_type.get("TextVectorization")
        embedding = by_type.get("Embedding")
        rnn = by_type.get("SimpleRNN")
        dropout = by_type.get("Dropout")

        vocab_size = int(vectorizer.vocabulary_size()) if vectorizer else 0
        seq_len = int(vectorizer.get_config().get("output_sequence_length", 0)) if vectorizer else 0
        embed_dim = int(embedding.output_dim) if embedding else 0
        rnn_units = int(rnn.units) if rnn else 0
        drop_rate = float(dropout.rate) if dropout else 0.0
        classes = int(self.model.output_shape[-1]) if self.model.output_shape else 3

        self.metadata = {
            "model_path": str(self.model_path),
            "architecture": "RNN",  # UI specification: display name is strictly RNN
            "vocabulary_size": vocab_size,
            "sequence_length": seq_len,
            "embedding_dim": embed_dim,
            "rnn_units": rnn_units,
            "dropout_rate": drop_rate,
            "classes": classes,
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, text: str) -> Dict[str, Any]:
        """Run sentiment prediction on a Vietnamese text comment.

        Args:
            text: Single input string.

        Returns:
            Dictionary containing prediction results:
            - text: trimmed input text
            - probabilities: list of 3 float probabilities [Negative, Neutral, Positive]
            - class_index: index of class with maximum probability (0, 1, or 2)
            - class_name: Vietnamese name of predicted class
            - class_name_en: English name of predicted class
            - confidence: maximum probability (float between 0 and 1)
            - elapsed_ms: inference time in milliseconds
        """
        if not self._is_loaded or self.model is None:
            raise PredictionError("Mô hình chưa được nạp. Hãy nạp mô hình trước khi dự đoán.")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Bình luận trống. Vui lòng nhập nội dung cần phân tích.")

        # Prepare 2D tensor string input (1, 1) as expected by the TextVectorization layer
        input_tensor = tf.constant([[cleaned_text]], dtype=tf.string)

        started = time.perf_counter()
        try:
            output = self.model(input_tensor, training=False)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            probs = np.asarray(output, dtype=np.float64)
            if probs.shape != (1, 3):
                raise PredictionError(f"Kích thước đầu ra không hợp lệ: cần (1, 3), nhận {probs.shape}")

            probs = probs[0]

            if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
                raise PredictionError("Xác suất mô hình trả về không hợp lệ (NaN hoặc số âm).")

            total_prob = probs.sum()
            if not np.isclose(total_prob, 1.0, atol=1e-4):
                raise PredictionError(f"Tổng ba xác suất không xấp xỉ 1 ({total_prob:.5f}).")

            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])

            return {
                "text": cleaned_text,
                "probabilities": [float(p) for p in probs],
                "class_index": class_idx,
                "class_name": LABEL_NAMES_VI[class_idx],
                "class_name_en": LABEL_NAMES_EN[class_idx],
                "confidence": confidence,
                "elapsed_ms": elapsed_ms,
            }
        except Exception as e:
            if isinstance(e, (PredictionError, ValueError)):
                raise
            raise PredictionError(f"Lỗi trong quá trình inference: {e}") from e
