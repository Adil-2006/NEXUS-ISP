"""Singleton model manager and warm-up utility."""
import logging
import time
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

    def initialize(self):
        if self._model is not None:
            return

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

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
