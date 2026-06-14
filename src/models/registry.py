"""Encoder registry and task head factory."""

import importlib
from types import SimpleNamespace
from typing import Literal

import torch.nn as nn

EncoderType = Literal[
    "TimesNet", "Autoformer", "FEDformer", "Informer",
    "NonStationary_Transformer", "Chronos", "Toto", "Mantis",
]

ENCODER_TYPES: tuple[EncoderType, ...] = (
    "TimesNet", "Autoformer", "FEDformer", "Informer",
    "NonStationary_Transformer", "Chronos", "Toto", "Mantis",
)


class EncoderRegistry:
    """Dynamically loads TelecomTS encoder models from src/encoders/.

    Each encoder module exports:
        class Model:
            def __init__(self, configs: SimpleNamespace)
            def forward(self, x_enc: Tensor[B,T,C]) -> Tensor[B,d_model]

    The encoders use relative imports (from encoders.utils.layers...),
    so src/ must be on sys.path for them to resolve.
    """

    @staticmethod
    def get_encoder(name: EncoderType, configs: dict) -> nn.Module:
        """Instantiate an encoder by name.

        Args:
            name: One of the 8 encoder types.
            configs: Dictionary of model hyperparameters
                     (e.g., d_model, e_layers, seq_len, enc_in, ...).

        Returns:
            The encoder nn.Module instance.

        Raises:
            ValueError: If encoder name is unknown.
        """
        if name not in ENCODER_TYPES:
            raise ValueError(
                f"Unknown encoder '{name}'. Available: {ENCODER_TYPES}"
            )

        module = importlib.import_module(f"encoders.{name}")
        model_cls = getattr(module, "Model")
        return model_cls(SimpleNamespace(**configs))

    @staticmethod
    def get_d_model(configs: dict) -> int:
        """Extract d_model from encoder config."""
        return configs.get("d_model", 32)
