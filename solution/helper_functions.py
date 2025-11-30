import numpy as np

def entropy_of_node(probabilities: list[float]) -> float:
    """Provides the entropy of a node given the class probabilities.

    Args:
        probabilities (list[float]): List of class probabilities for the node.

    Returns:
        float: Entropy value of the node.
    """
    if any(p == 0 for p in probabilities):
        return 0.0
    ret: float = 0.0
    for p in probabilities:
        ret += p * np.log2(p)
    ret *= -1
    return ret