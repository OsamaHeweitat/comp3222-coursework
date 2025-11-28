import numpy as np

def entropy_of_node(probabilities: list[float]) -> float:
    if any(p == 0 for p in probabilities):
        return 0.0
    ret: float = 0.0
    for p in probabilities:
        ret += p * np.log2(p)
    ret *= -1
    return ret