"""
ml_pipeline/datasets.py

Standardized dataset interfaces forwarding to authentic loaders in ml_pipeline/data/
"""

import sys
from pathlib import Path

# Add current directory to path if needed
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data.bigearthnet_loader import BigEarthNetDataset
from data.rsvqa_loader import RSVQADataset
from data.vrsbench_loader import VRSBenchDataset
from data.cdvqa_loader import CDVQADataset

__all__ = [
    "BigEarthNetDataset",
    "RSVQADataset",
    "VRSBenchDataset",
    "CDVQADataset",
]
