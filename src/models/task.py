from dataclasses import dataclass
import numpy as np


@dataclass(init=True)
class Task:
    step_id: int
    topic_id: int
    preprocessed_text: str
    raw_text: str
    vector: np.ndarray
