from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from solution.decision_stump import DecisionStumpClassifier
import numpy as np

class TreeEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators: int = 200, average_probas: bool = True, random_state: int | None = None):
        """ Initialize the Tree Ensemble Classifier.

        Args:
            n_estimators (int, optional): Controls ensemble size. Defaults to 200.
            average_probas (bool, optional): Switches between majority vote (hard voting) and averaging predicted probabilities (soft voting). Defaults to True.
            random_state (int | None, optional): Random seed for reproducibility. Defaults to None.
        """
        self.n_estimators = n_estimators
        self.average_probas = average_probas
        self.random_state = random_state
        self.estimators_: list[tuple[DecisionStumpClassifier, np.ndarray]] = []
        self.label_encoder_ = LabelEncoder()
    
    def fit(self, X, y) -> None:
        if np.issubdtype(X.dtype, np.floating):
            raise ValueError("No floats allowed in X.")

        X = self._normalise_X(X)
        y_encoded: np.ndarray = np.asarray(self.label_encoder_.fit_transform(y))

        self.classes_ = np.unique(y_encoded)
        self.n_classes_ = len(self.classes_)
        
        rng = np.random.RandomState(self.random_state)
        for estimator in range(self.n_estimators):
            seed = rng.randint(0, 2**32, dtype="uint32")
            estimator_rng = np.random.RandomState(seed)
            max_features = max(1, int(np.sqrt(X.shape[1])))
            feature_indices = estimator_rng.choice(X.shape[1], max_features, replace=False)
            row_indices: np.ndarray = estimator_rng.choice(X.shape[0], X.shape[0], replace=True)
            quality_measure_options = ["ig", "gain_ratio", "chi2", "chi2_yates"]
            quality_measure: str = estimator_rng.choice(quality_measure_options)
            stump_seed = estimator_rng.randint(0, 2**32, dtype="uint32")
            stump = DecisionStumpClassifier(n_attributes=None, quality_measure=quality_measure, random_state=stump_seed)
            stump.fit(X[row_indices][:, feature_indices], y_encoded[row_indices])
            self.estimators_.append((stump, feature_indices))
        self.n_features_in_ = X.shape[1]
        # self.describe()

    def describe(self):
        print("=== Tree Ensemble ===")
        print(f"n_estimators = {len(self.estimators_)}")
        print(f"average_probas = {self.average_probas}")

        for i, (stump, feature_indices) in enumerate(self.estimators_):
            print(f"\nEstimator {i}:")
            print(f"Feature subset used: {list(feature_indices)}")
            print("Chosen global feature:", feature_indices[stump.att_index])
            stump.describe()


    def predict_proba(self, X) -> np.ndarray:
        if np.issubdtype(X.dtype, np.floating):
            raise ValueError("No floats allowed in X.")

        X = self._normalise_X(X)

        n_samples = X.shape[0]
        n_classes = self.n_classes_
        
        proba_sum = np.zeros((n_samples, n_classes), dtype=float)

        for stump, feature_indices in self.estimators_:
            X_subset = X[:, feature_indices]
            member_proba = stump.predict_proba(X_subset)  # (n_samples, n_classes_stump)

            aligned = np.zeros((n_samples, n_classes), dtype=float)
            for i, cls in enumerate(stump.classes_):
                idx = np.where(self.classes_ == cls)[0][0]
                aligned[:, idx] = member_proba[:, i]

            proba_sum += aligned

        if self.average_probas:
            proba = proba_sum / len(self.estimators_)
        else:
            votes = np.argmax(proba_sum, axis=1)
            proba = np.zeros_like(proba_sum)
            proba[np.arange(n_samples), votes] = 1.0
        
        return proba

    def predict(self, X) -> np.ndarray:
        encoded = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(encoded)

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

    clf = TreeEnsembleClassifier(n_estimators=10, random_state=67)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Predictions:", y_pred)
    print("Accuracy:", accuracy_score(y, y_pred))