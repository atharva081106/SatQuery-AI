from model_interfaces import (
    SpecialistModel,
    SingleImageVQA,
    SingleImageCaptioning,
    SingleImageGrounding,
    BiTemporalChangeAnalysis,
    CrossModalAnalysis
)
from typing import Dict

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, SpecialistModel] = {}
        self.register(SingleImageVQA())
        self.register(SingleImageCaptioning())
        self.register(SingleImageGrounding())
        self.register(BiTemporalChangeAnalysis())
        self.register(CrossModalAnalysis())

    def register(self, model: SpecialistModel):
        self._models[model.task_name] = model

    def get_model(self, task_name: str) -> SpecialistModel:
        if task_name not in self._models:
            raise ValueError(f"No model registered for task: {task_name}")
        return self._models[task_name]

registry = ModelRegistry()
