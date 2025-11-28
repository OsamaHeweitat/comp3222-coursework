"""Put your functinos for assessing attribute quality here."""
import numpy as np
import solution.helper_functions as hf

def information_gain(table: np.ndarray) -> float:
    """Calculates the information gain of a given attribute split. 
    Can handle zero log calculations as they are safe-guarded against in the entropy_of_node function.
    Allows for any number of classes (columns) and any number of attribute values (rows).

    Args:
        table (np.ndarray): A 2D numpy array where rows represent attribute values and columns represent class counts. Example: np.array([[2, 3], [3, 2]]) 

    Returns:
        float: The information gain of the attribute split provided.
    """
    total_cases: int = np.sum(table)
    parent_probabilities: np.ndarray = np.sum(table, axis=0) / total_cases
    parent_entropy: float = hf.entropy_of_node(parent_probabilities.tolist())

    weighted_attribute_entropies: list[float] = []

    for row in table:
        row_total: int = np.sum(row) # case total for an attribute (e.g. yes peaty or no peaty)
        row_probabilities: np.ndarray = row / row_total
        row_entropy: float = hf.entropy_of_node(row_probabilities.tolist())
        weighted_attribute_entropies.append((row_total / total_cases) * row_entropy)
            
    info_gain: float = parent_entropy - sum(weighted_attribute_entropies)

    return info_gain

def information_gain_ratio(table: np.ndarray) -> float:
    """Calculates the information gain ratio of a given attribute split.
    Can handle zero log calculations as they are safe-guarded against in the information_gain function and in the split info calculation.
    Allows for any number of classes (columns) and any number of attribute values (rows).

    Args:
        table (np.ndarray): A 2D numpy array where rows represent attribute values and columns represent class counts. Example: np.array([[2, 3], [3, 2]])

    Returns:
        float: The information gain ratio of the attribute split provided.
    """
    gain: float = information_gain(table)
    
    weighted_splits: list[float] = []
    total_cases: int = np.sum(table)
    for row in table:
        if np.sum(row) == 0:
            continue
        weighted_splits.append((np.sum(row) / total_cases) * np.log2(np.sum(row) / total_cases))
    
    split_info: float = -sum(weighted_splits)
    ratio = gain / split_info if split_info != 0 else 0
    return ratio

def chi_squared(table: np.ndarray) -> float:
    """Calculates the chi-squared statistic of a given attribute split.
    Can handle any number of classes (columns) and any number of attribute values (rows).
    Safe-guards against division by zero in expected count calculation.

    Args:
        table (np.ndarray): A 2D numpy array where rows represent attribute values and columns represent class counts. Example: np.array([[2, 3], [3, 2]])

    Returns:
        float: The chi-squared statistic of the attribute split provided.
    """
    chi_squared: float = 0.0
    total_cases: int = np.sum(table)
    parent_probabilities: np.ndarray = np.sum(table, axis=0) / total_cases

    for row in table:
        row_total: int = np.sum(row)
        for column in range(len(row)):
            expected_count: float = row_total * parent_probabilities[column]
            if expected_count > 0:
                chi_squared += (row[column] - expected_count) ** 2 / expected_count
    return chi_squared

def chi_squared_yates(table: np.ndarray) -> float:
    """Calculates the chi-squared statistic with Yates' correction of a given attribute split.
    Can handle any number of classes (columns) and any number of attribute values (rows).
    Safe-guards against division by zero in expected count calculation.

    Args:
        table (np.ndarray): A 2D numpy array where rows represent attribute values and columns represent class counts. Example: np.array([[2, 3], [3, 2]])

    Returns:
        float: The chi-squared statistic with Yates' correction of the attribute split provided.
    """
    chi_squared_yates: float = 0.0
    total_cases: int = np.sum(table)
    parent_probabilities: np.ndarray = np.sum(table, axis=0) / total_cases

    for row in table:
        row_total: int = np.sum(row)
        for column in range(len(row)):
            expected_count: float = row_total * parent_probabilities[column]
            if expected_count > 0:
                chi_squared_yates += ((abs(row[column] - expected_count) - 0.5) ** 2) / expected_count
    return chi_squared_yates

if __name__ == "__main__":
    peaty_split = np.array([[4, 0], [1, 5]])
    print("Peaty Split from Root: IG: ", round(information_gain(peaty_split), 3), " Chi2: ", round(chi_squared(peaty_split), 3))

    woody_split = np.array([[2, 3], [3, 2]])
    print("Woody Split from Root: IG: ", round(information_gain(woody_split), 3), " Chi2: ", round(chi_squared(woody_split), 3))

    sweet_split = np.array([[2, 5], [3, 0]])
    print("Sweet Split from Root: IG: ", round(information_gain(sweet_split), 3), " Chi2: ", round(chi_squared(sweet_split), 3))

    peaty_woody_split = np.array([[1, 3], [0, 2]])
    print("Woody Split from Not Peaty: IG: ", round(information_gain(peaty_woody_split), 3), " Chi2: ", round(chi_squared(peaty_woody_split), 3))

    peaty_sweet_split = np.array([[0, 5], [1, 0]])
    print("Sweet Split from Not Peaty: IG: ", round(information_gain(peaty_sweet_split), 3), " Chi2: ", round(chi_squared(peaty_sweet_split), 3))    