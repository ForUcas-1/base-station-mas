"""Task-specific linear head factory.

Mirrors TelecomTS train_utils.prepare() head creation pattern.
All heads take (B, d_model) as input and produce task-specific output.
"""

import torch.nn as nn


def build_head(
    task_type: str,
    d_model: int,
    seq_len: int = 128,
    enc_in: int = 18,
    dropout: float = 0.2,
) -> nn.Sequential:
    """Build a task-specific prediction head.

    Args:
        task_type: One of "anomaly detection", "root-cause analysis",
                   "anomaly duration", "forecasting".
        d_model: Encoder output dimension (must match the encoder's d_model).
        seq_len: Sequence length (used by anomaly duration head).
        enc_in: Number of input channels (used by forecasting head).
        dropout: Dropout probability for root-cause head.

    Returns:
        A nn.Sequential head module.

    Raises:
        ValueError: If task_type is unknown.
    """
    task_type = task_type.lower().strip()

    if task_type == "anomaly detection":
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    elif task_type == "root-cause analysis":
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 10),  # 10 classes (checkpoint trained without Jamming)
        )

    elif task_type == "anomaly duration":
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, seq_len),
            nn.Sigmoid(),
        )

    elif task_type == "forecasting":
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, enc_in),
        )

    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Available: anomaly detection, root-cause analysis, "
            f"anomaly duration, forecasting"
        )
