"""TelecomTS dataset loader.

Loads the TelecomTS dataset from local Arrow cache or HuggingFace Hub.
"""

import glob
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets
from datasets import config as hf_config


class DatasetLoader:
    """Loads TelecomTS dataset with local cache support.

    The dataset contains 32,000 time-series samples with 18 KPI channels
    at 128 timesteps each. Each sample includes text fields (description,
    anomalies, labels, QnA, troubleshooting_tickets, statistics).

    Usage:
        loader = DatasetLoader()
        sample = loader[0]         # Get single sample by index
        train, test = loader.split(test_size=0.2)
    """

    DATASET_NAME = "AliMaatouk/TelecomTS"
    NUM_SAMPLES = 32000

    def __init__(self, cache_dir: str | None = None):
        """
        Args:
            cache_dir: Override HF datasets cache directory.
                       Defaults to DATA_CACHE_DIR env or 'data/cache/'.
        """
        if cache_dir is None:
            cache_dir = os.environ.get(
                "DATA_CACHE_DIR",
                str(Path(__file__).resolve().parent.parent.parent / "data" / "cache"),
            )

        self.cache_dir = Path(cache_dir)
        self._dataset: Dataset | None = None

    @property
    def dataset(self) -> Dataset:
        """Lazy-load the full dataset (32K samples) from local Arrow cache."""
        if self._dataset is None:
            self._dataset = self._load_from_cache()
        return self._dataset

    def _load_from_cache(self) -> Dataset:
        """Load dataset directly from cached Arrow files.

        Bypasses the HuggingFace Hub config hash resolution that can fail
        when the datasets library version changes. Reads Arrow shards directly.
        """
        # Find all Arrow shard files in the cache tree
        arrow_pattern = str(
            self.cache_dir / "AliMaatouk___telecom_ts" / "**" / "*.arrow"
        )
        arrow_files = sorted(glob.glob(arrow_pattern, recursive=True))

        if arrow_files:
            # Load each shard and concatenate
            shards = [
                Dataset.from_file(f, split="full") for f in arrow_files
            ]
            return concatenate_datasets(shards)

        # Fallback: try HuggingFace Hub
        try:
            hf_config.HF_DATASETS_CACHE = str(self.cache_dir)
            from datasets import load_dataset
            return load_dataset(
                self.DATASET_NAME,
                data_files={"full": "**/chunked.jsonl"},
                split="full",
                trust_remote_code=False,
            )
        except Exception:
            from datasets import load_dataset
            return load_dataset(
                self.DATASET_NAME,
                data_files={"full": "**/chunked.jsonl"},
            )["full"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single sample by index (0-31999)."""
        return self.dataset[int(index)]

    def get_sample(self, index: int) -> dict[str, Any]:
        """Get a single sample with all fields."""
        return self[int(index)]

    def split(
        self,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> tuple[Dataset, Dataset]:
        """Return (train_dataset, test_dataset)."""
        split_dict = self.dataset.train_test_split(
            test_size=test_size,
            seed=seed,
        )
        return split_dict["train"], split_dict["test"]

    def sample_with_anomaly(self, anomaly_type: str | None = None) -> dict[str, Any]:
        """Find a sample that has an anomaly, optionally of a specific type.

        Args:
            anomaly_type: Filter by anomaly type (e.g., "Jamming").
                          If None, returns any sample with anomalies.

        Returns:
            A dataset sample dict.

        Raises:
            StopIteration: If no matching sample is found.
        """
        for i in range(len(self)):
            sample = self[i]
            anomalies = sample.get("anomalies", {})
            if not anomalies or not anomalies.get("type"):
                continue
            if anomaly_type is None:
                return sample
            if anomalies["type"] == anomaly_type:
                return sample
        raise StopIteration(f"No sample found with anomaly_type='{anomaly_type}'")

    def sample_without_anomaly(self) -> dict[str, Any]:
        """Find a normal (no-anomaly) sample."""
        for i in range(len(self)):
            sample = self[i]
            anomalies = sample.get("anomalies", {})
            if not anomalies or not anomalies.get("type"):
                return sample
        raise StopIteration("No normal sample found")

    def iter_samples(self, start: int = 0, end: int | None = None):
        """Yield samples in range [start, end)."""
        end = end or len(self)
        for i in range(start, end):
            yield self[i]
