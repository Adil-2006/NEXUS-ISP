"""Singleton model manager and warm-up utility with cloud deployment checkpoint hydration."""
import os
import base64
import logging
import time
import urllib.request
import torch
import numpy as np
from pathlib import Path
from app.core.config import settings
from app.isp.models import load_ckpt, count_params, HybridISP

logger = logging.getLogger("nexus_isp.model_loader")

class ModelManager:
    _instance = None
    _model = None
    _metadata = None
    _device = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_checkpoint_exists(self, model_path: Path):
        """If the checkpoint is not present on disk, hydrate it from environment, URL, or embedded fallback."""
        if model_path.exists():
            return

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint not found at {model_path}. Attempting cloud hydration...")

        # 1. Check for Base64 encoded weights in environment
        b64_data = os.getenv("MODEL_BASE64")
        if b64_data:
            logger.info("Hydrating checkpoint from MODEL_BASE64 environment variable...")
            try:
                raw_bytes = base64.b64decode(b64_data.strip())
                model_path.write_bytes(raw_bytes)
                logger.info(f"Successfully restored checkpoint from MODEL_BASE64 ({len(raw_bytes):,} bytes).")
                return
            except Exception as e:
                logger.error(f"Failed to decode MODEL_BASE64: {e}")

        # 2. Check for public URL download (e.g. GitHub Releases, HuggingFace)
        model_url = os.getenv("MODEL_URL")
        if model_url:
            logger.info(f"Downloading checkpoint from MODEL_URL: {model_url}...")
            try:
                urllib.request.urlretrieve(model_url, str(model_path))
                logger.info("Successfully downloaded checkpoint from MODEL_URL.")
                return
            except Exception as e:
                logger.error(f"Failed to download checkpoint from MODEL_URL: {e}")

        # 3. Automatic embedded fallback (zero-configuration cloud deployments)
        try:
            from app.core.embedded_weights import EMBEDDED_MODEL_BASE64
            if EMBEDDED_MODEL_BASE64:
                logger.info("Hydrating checkpoint from embedded fallback weights...")
                raw_bytes = base64.b64decode(EMBEDDED_MODEL_BASE64.strip())
                model_path.write_bytes(raw_bytes)
                logger.info(f"Successfully restored checkpoint from embedded weights ({len(raw_bytes):,} bytes).")
                return
        except Exception as e:
            logger.warning(f"Could not load embedded fallback weights: {e}")

        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. "
            "Please ensure the checkpoint exists, set the MODEL_BASE64 environment variable, "
            "or set the MODEL_URL environment variable."
        )

    def initialize(self):
        if self._model is not None:
            return

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = Path(settings.MODEL_PATH)
        self._ensure_checkpoint_exists(model_path)

        logger.info(f"Loading Hybrid ISP checkpoint from: {model_path} onto {self._device}...")
        t0 = time.perf_counter()
        self._model, ck = load_ckpt(str(model_path), device=self._device)
        load_time_ms = (time.perf_counter() - t0) * 1e3
        
        num_params = count_params(self._model)
        self._metadata = {
            "mode": ck.get("mode", "hybrid"),
            "base": ck.get("base", 16),
            "depth": ck.get("depth", 3),
            "sep": ck.get("sep", True),
            "params": num_params,
            "params_k": round(num_params / 1e3, 2),
            "device": self._device,
            "load_time_ms": round(load_time_ms, 2),
            "checkpoint_path": str(model_path)
        }
        logger.info(f"Loaded {self._metadata['mode']} model with {num_params:,} parameters in {load_time_ms:.2f} ms")

        # Warm-up inference
        dummy_front = torch.zeros((1, 3, 128, 128), dtype=torch.float32, device=self._device)
        dummy_sigma = torch.tensor([0.02], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            _ = self._model(dummy_front, dummy_sigma)
        logger.info("Model warm-up complete.")

    @property
    def model(self) -> HybridISP:
        if self._model is None:
            self.initialize()
        return self._model

    @property
    def device(self) -> str:
        if self._device is None:
            self.initialize()
        return self._device

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self.initialize()
        return self._metadata

model_manager = ModelManager.get_instance()
