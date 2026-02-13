from typing import Dict, Optional

from methods.api import RecourseMethod
from models.api import MLModel


class Greedy(RecourseMethod):
    """
    TensorFlow-based Greedy implementation has been deprecated.
    """

    _DEFAULT_HYPERPARAMS: Dict = {}

    def __init__(self, mlmodel: MLModel = None, hyperparams: Optional[Dict] = None):
        raise RuntimeError("Greedy is disabled because TensorFlow support was removed.")

