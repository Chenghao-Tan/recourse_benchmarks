from typing import Dict

from methods.api import RecourseMethod


class CEM(RecourseMethod):
    """
    TensorFlow-based CEM implementation has been deprecated.
    """

    _DEFAULT_HYPERPARAMS: Dict = {}

    def __init__(self, *args, **kwargs):
        raise RuntimeError("CEM is disabled because TensorFlow support was removed.")
