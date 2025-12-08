import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, roc_curve, auc, recall_score
from solution.tree_ensemble import TreeEnsembleClassifier
from provided_code.data_loaders import load_tabular_xy
import time
import pandas as pd
import matplotlib.pyplot as plt

dirs = [
    "balance-scale",
    # "chess-krvk",
    "chess-krvkp",
    # "connect-4",
    "contraceptive-method",
    "fertility",
    "habermans-survival",
    "hayes-roth",
    "led-display",
    # "lymphography",
    "molecular-promoters",
    # "molecular-splice",
    "monks-1",
    "monks-2",
    "monks-3",
    # "nursery",
    "optdigits",
    # "pendigits",
    # "semeion",
    "spect-heart",
    "tic-tac-toe",
    # "zoo",
]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=67)

def evaluate_tree_native(X, y):
    scores = []
    y_true_all = []
    y_pred_all = []
    proba_all = []
    for train_id, test_id in cv.split(X, y):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]

        clf = TreeEnsembleClassifier(n_estimators=80, random_state=67)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)
        scores.append(accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test.tolist())  
        y_pred_all.extend(y_pred.tolist())
        proba_all.extend(proba.tolist())

    return np.array(scores), np.array(y_true_all), np.array(y_pred_all), np.array(proba_all)

def evaluate_tree_oneshot(X, y):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(X)
    scores = []
    y_true_all = []
    y_pred_all = []
    proba_all = []
    for train_id, test_id in cv.split(X, y):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]

        X_train_enc = enc.transform(X_train).astype(np.int32) # type: ignore
        X_test_enc = enc.transform(X_test).astype(np.int32) # type: ignore

        clf = TreeEnsembleClassifier(n_estimators=80, random_state=67)
        clf.fit(X_train_enc, y_train)
        y_pred = clf.predict(X_test_enc)
        proba = clf.predict_proba(X_test_enc)  

        scores.append(accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test.tolist())  
        y_pred_all.extend(y_pred.tolist())
        proba_all.extend(proba.tolist())

    return np.array(scores), np.array(y_true_all), np.array(y_pred_all), np.array(proba_all)

def evaluate_rf_oneshot(X, y):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(X)
    scores = []
    y_true_all = []
    y_pred_all = []
    proba_all = []
    for train_id, test_id in cv.split(X, y):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]

        X_train_enc = enc.transform(X_train).astype(np.int32) # type: ignore
        X_test_enc = enc.transform(X_test).astype(np.int32) # type: ignore

        clf = RandomForestClassifier(n_estimators=80, random_state=67)
        clf.fit(X_train_enc, y_train)
        y_pred = clf.predict(X_test_enc)
        proba = clf.predict_proba(X_test_enc)  

        scores.append(accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test.tolist())  
        y_pred_all.extend(y_pred.tolist())
        proba_all.extend(proba.tolist())

    return np.array(scores), np.array(y_true_all), np.array(y_pred_all), np.array(proba_all)

if __name__ == "__main__":
    all_results = {
        "dataset": [],
        "tree_native": [],
        "tree_ohe": [],
        "rf_ohe": []
    }
    y_true_all = []
    y_pred_native_all = []
    X_all = None
    ensemble_native_final = None
    proba_native_final = None
    proba_tree_ohe_final = None
    proba_rf_oneshot_final = None
    for name in dirs:
        X, y= load_tabular_xy("./data/"+name+"/"+name+".data")
        print(name+":", type(X), X.shape, y.shape, "Unique y:", set(y))

        time_start = time.time()
        tree_native_scores, y_true_tree_native, y_pred_tree_native, proba_tree_native = evaluate_tree_native(X, y)
        time_end = time.time()
        print(f"  Native took {time_end - time_start:.2f} seconds")
        time_start = time.time()
        tree_oneshot_scores, y_true_tree_oneshot, y_pred_tree_oneshot, proba_tree_oneshot = evaluate_tree_oneshot(X, y)
        time_end = time.time()
        print(f"  OneHot took {time_end - time_start:.2f} seconds")
        time_start = time.time()
        rf_oneshot_scores, y_true_rf_oneshot, y_pred_rf_oneshot, proba_rf_oneshot = evaluate_rf_oneshot(X, y)
        time_end = time.time()
        print(f"  RF OneHot took {time_end - time_start:.2f} seconds")

        print(f"  Tree Native Accuracy: {np.mean(tree_native_scores):.4f} ± {np.std(tree_native_scores):.4f}")
        print(f"  Tree OneHot Accuracy: {np.mean(tree_oneshot_scores):.4f} ± {np.std(tree_oneshot_scores):.4f}")
        print(f"  RF OneHot Accuracy:   {np.mean(rf_oneshot_scores):.4f} ± {np.std(rf_oneshot_scores):.4f}")

        all_results["dataset"].append(name)
        all_results["tree_native"].append(np.mean(tree_native_scores))
        all_results["tree_ohe"].append(np.mean(tree_oneshot_scores))
        all_results["rf_ohe"].append(np.mean(rf_oneshot_scores))

        if name == "optdigits":
            y_true_all = y_true_tree_native
            y_pred_native_all = y_pred_tree_native
            y_pred_ohe_all = y_pred_tree_oneshot
            y_pred_rf_ohe_all = y_pred_rf_oneshot
            proba_native_final = proba_tree_native
            proba_tree_ohe_final = proba_tree_oneshot
            proba_rf_oneshot_final = proba_rf_oneshot

    y_true_all = np.array(y_true_all)
    y_pred_native_all = np.array(y_pred_native_all)

    df = pd.DataFrame(all_results)
    print(df)

    plt.figure(figsize=(8,6))
    plt.scatter(df["tree_native"], df["tree_ohe"])

    lims = [
        min(df["tree_native"].min(), df["tree_ohe"].min()),
        max(df["tree_native"].max(), df["tree_ohe"].max()),
    ]
    plt.plot(lims, lims, 'k--')

    for i, row in df.iterrows():
        plt.annotate(row["dataset"], (row["tree_native"], row["tree_ohe"]), fontsize=8)

    plt.xlabel("TreeEnsemble Native Accuracy")
    plt.ylabel("TreeEnsemble One-Hot Accuracy")
    plt.title("RQ1 - Native vs One-Hot Encoding")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(10,6))
    plt.bar(x - width, df["tree_native"], width, label="Tree Native")
    plt.bar(x,         df["tree_ohe"],    width, label="Tree OneHot")
    plt.bar(x + width, df["rf_ohe"],      width, label="RF OneHot")

    plt.xticks(x, df["dataset"].astype(str).tolist(), rotation=90)
    plt.ylabel("Accuracy")
    plt.title("RQ2 - Classifier Comparison Across Datasets")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8,8))
    ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_native_all, ax=ax)
    plt.title("Optdigits - Confusion Matrix (Tree Native)")
    plt.show()

    fig, ax = plt.subplots(figsize=(8,8))
    ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_ohe_all, ax=ax)
    plt.title("Optdigits - Confusion Matrix (Tree OneHot)")
    plt.show()

    fig, ax = plt.subplots(figsize=(8,8))
    ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_rf_ohe_all, ax=ax)
    plt.title("Optdigits - Confusion Matrix (RF OneHot)")
    plt.show()

    classes = np.unique(y_true_all)
    y_bin = label_binarize(y_true_all, classes=classes)

    proba = proba_native_final
    
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i]) # type: ignore
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label=f"Class {cls} (AUC={score:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Optdigits - ROC Curves (Tree Native)")
    plt.legend()
    plt.show()

    proba = proba_tree_ohe_final

    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i]) # type: ignore
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label=f"Class {cls} (AUC={score:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Optdigits - ROC Curves (Tree OneHot)")
    plt.legend()
    plt.show()

    proba = proba_rf_oneshot_final
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i]) # type: ignore
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label=f"Class {cls} (AUC={score:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Optdigits - ROC Curves (RF OneHot)")
    plt.legend()
    plt.show()

    balanced_per_class = recall_score(y_true_all, y_pred_native_all, average=None)

    plt.figure(figsize=(8,5))
    plt.bar(classes, balanced_per_class)
    plt.xlabel("Digit Class")
    plt.ylabel("Balanced Accuracy")
    plt.title("Optdigits - Per-Class Balanced Accuracy (Tree Native)")
    plt.show()

    balanced_per_class = recall_score(y_true_all, y_pred_ohe_all, average=None)
    plt.figure(figsize=(8,5))
    plt.bar(classes, balanced_per_class)
    plt.xlabel("Digit Class")
    plt.ylabel("Balanced Accuracy")
    plt.title("Optdigits - Per-Class Balanced Accuracy (Tree OneHot)")
    plt.show()

    balanced_per_class = recall_score(y_true_all, y_pred_rf_ohe_all, average=None)
    plt.figure(figsize=(8,5))
    plt.bar(classes, balanced_per_class)
    plt.xlabel("Digit Class")
    plt.ylabel("Balanced Accuracy")
    plt.title("Optdigits - Per-Class Balanced Accuracy (RF OneHot)")
    plt.show()
