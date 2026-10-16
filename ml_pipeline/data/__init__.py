"""
ml_pipeline/data package initialization
Exports loaders for all 4 authentic remote sensing datasets:
1. BigEarthNet.txt (arXiv:2603.29630)
2. RSVQA (arXiv:2003.07333)
3. VRSBench (arXiv:2406.12384)
4. CDVQA (arXiv:2112.06343)
"""

from .bigearthnet_loader import BigEarthNetDataset, CLC_19_CLASSES
from .rsvqa_loader import RSVQADataset, QUESTION_TYPES as RSVQA_QUESTION_TYPES
from .vrsbench_loader import VRSBenchDataset
from .cdvqa_loader import CDVQADataset, CHANGE_QUESTION_TYPES

__all__ = [
    "BigEarthNetDataset",
    "CLC_19_CLASSES",
    "RSVQADataset",
    "RSVQA_QUESTION_TYPES",
    "VRSBenchDataset",
    "CDVQADataset",
    "CHANGE_QUESTION_TYPES",
]
