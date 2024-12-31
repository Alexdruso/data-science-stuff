import numpy as np
from typing import Any


def compare_signatures(A: np.ndarray[Any, Any], B: np.ndarray[Any, Any]) -> float:
    """
    Estimates the Jaccard similarity between A and B as the probability to have an equal entry in the signatures of A
    and B.
    :param A: the minhashed representation of document A
    :param B: the minhashed representation of document A
    :return: an estimate of the Jaccard similarity between A and B
    """
    return np.sum(A == B) / len(A)
