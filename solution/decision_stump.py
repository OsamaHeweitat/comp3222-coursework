"""Skeleton code for Decision Stump implementation for part 2,1."""


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import accuracy_score
import numpy as np

class DecisionStumpClassifier(BaseEstimator, ClassifierMixin):
    """ A Decision Stump Classifier that selects the best attribute to split on based on a specified quality measure.

    Args:
        BaseEstimator (_type_): _description_
        ClassifierMixin (_type_): _description_
    """

    def __init__(self, n_attributes: int | None = None, quality_measure: str = "ig", random_state: int | None = None, alpha: float = 1.0) -> None:
        """ Initialize the Decision Stump Classifier.

        Args:
            n_attributes (int | None, optional): The number of (randomly selected) attributes to evaluate when finding the best split. Defaults to None which evaluates all.
            quality_measure (str, optional): The quality measure to use for evaluating splits. Defaults to "ig", options are "ig", "gain_ratio", "chi2", "chi2_yates".
            random_state (int | None, optional): The random seed for attribute selection. Defaults to None.
            alpha (float, optional): The alpha used for laplace smoothing. Defaults to 1.0.
        
        Returns:
            None: This method does not return any value.
        """
        self.n_attributes = n_attributes
        self.quality_measure = quality_measure # "ig", "gain_ratio", "chi2", "chi2_yates"
        self.random_state = random_state
        self.alpha = alpha

    def fit(self, X, y) -> None:
        """
        Fit the Decision Stump Classifier to the data (Classifier training logic). 
        Performs a tie-breaker by choosing the first attribute in case of a tie in quality measure.
        Handles missing values by treating them as a separate category using normalisation.
        
        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_attributes).
            y (np.ndarray): Target labels of shape (n_samples).
        Raises:
            ValueError: If X contains floating point values.
        
        Returns:
            None: This method does not return any value.
        """
        if np.issubdtype(X.dtype, np.floating):
            raise ValueError("No floats allowed in X.")
        
        X = self._normalise_X(X)
        
        if self.n_attributes is None:
            attributes_to_evaluate = np.arange(X.shape[1])
        else:
            rng = np.random.RandomState(self.random_state)
            attributes_to_evaluate = rng.choice(X.shape[1], self.n_attributes, replace=False)
        
        classes = np.unique(y)
        self.classes_ = classes
        c = len(classes)

        class_lookup = {cls: idx for idx, cls in enumerate(classes)}

        chosen_attr: int = -1
        chosen_attr_quality: float = -1.0
        for attr in attributes_to_evaluate:
            col: np.ndarray = X[:, attr]
            unique_values: np.ndarray = np.unique(col)
            v: int = len(unique_values)
            table: np.ndarray = np.zeros((v, c), dtype=int)
            for i in range(X.shape[0]):
                # row_index = np.where(unique_values == X[i, attr])[0][0]
                # class_index: int = np.where(classes == y[i])[0][0]
                row_index = np.where(unique_values == X[i, attr])[0][0]
                class_index = class_lookup[y[i]]
                table[row_index, class_index] += 1
            attr_quality: float = 0.0

            match self.quality_measure:
                case "ig":
                    from solution.attribute_quality import information_gain
                    attr_quality = information_gain(table)
                case "gain_ratio":
                    from solution.attribute_quality import information_gain_ratio
                    attr_quality = information_gain_ratio(table)
                case "chi2":
                    from solution.attribute_quality import chi_squared
                    attr_quality = chi_squared(table)
                case "chi2_yates":
                    from solution.attribute_quality import chi_squared_yates
                    attr_quality = chi_squared_yates(table)

            if attr_quality >= chosen_attr_quality:
                chosen_attr_quality = attr_quality
                chosen_attr = attr
                chosen_values = unique_values
                chosen_table = table

        self.att_index = chosen_attr
        self.children_counts = {}
        for i, val in enumerate(chosen_values):
            # self.children_counts[val] = chosen_table[i, :]
            full_row = np.zeros(len(classes), dtype=int)
            full_row[:chosen_table[i].shape[0]] = chosen_table[i]
            self.children_counts[val] = full_row
        # self.root_counts = np.bincount(y, minlength=c)
        self.root_counts = np.bincount(y, minlength=len(classes))
            
    def predict_proba(self, X) -> np.ndarray:
        """ Predict class probabilities for X.
        Has fallback to root prior with Laplace smoothing for unseen categories.
        Performs Laplace smoothing for all predictions.
        Handles missing values by treating them as a separate category using normalisation.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_attributes).

        Raises:
            ValueError: If X contains floating point values.

        Returns:
            np.ndarray: Predicted class probabilities of shape (n_samples, n_classes).
        """
        if np.issubdtype(X.dtype, np.floating):
            raise ValueError("No floats allowed in X.")
        X = self._normalise_X(X)
        n_samples = X.shape[0]
        n_classes = len(self.root_counts)
        probs = np.zeros((n_samples, n_classes), dtype=float)
        for i in range(n_samples):
            attr_value = X[i, self.att_index]
            if attr_value in self.children_counts:
                class_counts = self.children_counts[attr_value]
            else:
                class_counts = self.root_counts
            # (count + alpha) / (N + alpha*C)
            laplace_smoothed = class_counts + self.alpha
            laplace_smoothed = laplace_smoothed / np.sum(laplace_smoothed)
            probs[i] = laplace_smoothed
        
        return probs

    def predict(self, X) -> np.ndarray:
        """ Predict class labels for X.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_attributes).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def _normalise_X(self, X) -> np.ndarray:
        """ Normalises the input feature matrix X by replacing None, empty strings, and NaN values with a placeholder string "__MISSING__".

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_attributes).

        Returns:
            np.ndarray: Normalised feature matrix of shape (n_samples, n_attributes).
        """
        X_norm = X.astype(object).copy()

        X_norm[X_norm == None] = "__MISSING__"

        X_norm[X_norm == ""] = "__MISSING__"

        nan_mask = np.vectorize(lambda v: isinstance(v, float) and np.isnan(v))(X_norm)
        X_norm[nan_mask] = "__MISSING__"
        return X_norm

    def describe(self):
        print(f"--- Decision Stump ---")
        # print(f"Chosen attribute index: {self.att_index}")
        print(f"Quality measure: {self.quality_measure}")
        print(f"Classes: {self.classes_}")
        print("Root counts:", self.root_counts)

        print("Children:")
        for val, counts in self.children_counts.items():
            print(f"  If X[{self.att_index}] == {val}: class_counts = {counts}")

if __name__ == "__main__":
    X = np.array([
        ["yes", "no",  "yes"],
        ["yes", "yes", "yes"],
        ["yes", "no",  "no"],
        ["yes", "no",  "no"],
        ["no",  "yes", "no"],
        ["no",  "yes", "yes"],
        ["no",  "yes", "yes"],
        ["no",  "yes", "yes"],
        ["no",  "no",  "yes"],
        ["no",  "no",  "yes"],
    ], dtype=object)

    y = np.array([
        "Islay",
        "Islay",
        "Islay",
        "Islay",
        "Islay",
        "Speyside",
        "Speyside",
        "Speyside",
        "Speyside",
        "Speyside",
    ], dtype=object)

    classes, y_encoded = np.unique(y, return_inverse=True)

    for measure in ["ig", "gain_ratio", "chi2", "chi2_yates"]:
        stump = DecisionStumpClassifier(
            n_attributes=None,
            quality_measure=measure,
            alpha=1.0
        )

        stump.fit(X, y_encoded)
        preds = stump.predict(X)
        # print(f"Predictions using measure {measure}: {[classes[p] for p in preds]}")

        acc = accuracy_score(y_encoded, preds)

        print(f"DT using measure {stump.quality_measure} on whisky problem has accuracy {acc:.3f}")