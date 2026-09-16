import argparse
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, roc_auc_score, roc_curve, auc

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument(
    "--include-perplexity",
    action="store_true",
    default=False,
    help="Keep the perplexity column instead of dropping it (default: dropped).",
)
_cli_args, _ = _arg_parser.parse_known_args()

csv_path = "/Users/sylviadong/Documents/new_merged.csv"
df = pd.read_csv(csv_path)

# Directory to save ROC/AUC plots. Edit this path to where you want the PNGs saved.
OUTPUT_DIR = "/Users/sylviadong/Documents/roc_plots"

# Drop label and any non-feature columns (drop if present)
_cols_to_drop = ["group", "file", "single_text_score"]
if not _cli_args.include_perplexity:
    _cols_to_drop.append("perplexity")
drop_cols = [c for c in _cols_to_drop if c in df.columns]
X = df.drop(columns=drop_cols)
y = df["group"].map({"Human": 0, "AI": 1})

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.80,   # 80% test, 20% train
    random_state=1234,
    stratify=y
)

# Standardize features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression (L1)
alpha = 0.0001           # Regularization strength (you can change this)
C = 1 / alpha         # sklearn uses C = 1 / alpha

logreg = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=C,
    max_iter=1000,
    random_state=1234
)

# Gaussian Naive Bayes
gnb = GaussianNB()

# Decision Tree
dt = DecisionTreeClassifier(max_depth= 5, random_state=1234)
# Support Vector Machine (use probability=True so we can compute ROC AUC)
svm = SVC(kernel='rbf', probability=True, random_state=1234)

# Random Forest
rf = RandomForestClassifier(max_depth = 5, n_estimators=100, random_state=1234)

# Multilayer Perceptron (Neural Network)
nn = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                    max_iter=500, early_stopping=True, random_state=1234)

models = [
    ("Logistic L1", logreg),
    ("GaussianNB", gnb),
    ("Decision Tree", dt),
    ("SVM (RBF)", svm),
    ("Random Forest", rf),
    ("MLP (NN)", nn),
]


def fit_and_eval(name, clf, X_tr, X_te, y_tr, y_te):
    clf.fit(X_tr, y_tr)
    y_tr_pred = clf.predict(X_tr)
    y_te_pred = clf.predict(X_te)
    tr_acc = accuracy_score(y_tr, y_tr_pred)
    te_acc = accuracy_score(y_te, y_te_pred)
    tr_kappa = None
    te_kappa = None
    try:
        tr_kappa = cohen_kappa_score(y_tr, y_tr_pred)
    except Exception:
        tr_kappa = None
    try:
        te_kappa = cohen_kappa_score(y_te, y_te_pred)
    except Exception:
        te_kappa = None
    auc = None
    if hasattr(clf, "predict_proba"):
        try:
            y_te_prob = clf.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_te_prob)
        except Exception:
            auc = None
    return {
        "name": name,
        "train_acc": tr_acc,
        "test_acc": te_acc,
        "train_kappa": tr_kappa,
        "test_kappa": te_kappa,
        "auc": auc,
        "model": clf,
    }


results = []
for name, clf in models:
    # use scaled features for models that benefit from scaling (SVM, Logistic, GNB); Decision Tree/RandomForest are scale-invariant, but using scaled data is fine
    res = fit_and_eval(name, clf, X_train_scaled, X_test_scaled, y_train, y_test)
    results.append(res)


# Print learned weights for logistic separately
if hasattr(logreg, 'coef_'):
    coefficients = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    print("\n=== Learned Metric Weights (Logistic L1) ===\n")
    for feature, weight in zip(X.columns, coefficients):
        print(f"{feature}: {weight:.6f}")
    print(f"\nIntercept: {intercept:.6f}")


print("\n=== Model Accuracies ===")
for r in results:
    print(f"\n{r['name']}")
    print(f"  Training Accuracy: {r['train_acc']:.4f}")
    print(f"  Test Accuracy:     {r['test_acc']:.4f}")
    if r['auc'] is not None:
        print(f"  Test ROC AUC:      {r['auc']:.4f}")
    # print kappa values if available
    if r.get('train_kappa') is not None:
        print(f"  Training Cohen's Kappa: {r['train_kappa']:.4f}")
    if r.get('test_kappa') is not None:
        print(f"  Test Cohen's Kappa:     {r['test_kappa']:.4f}")


# --- ROC curve plotting for each model and a combined plot ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# compute and save individual ROC plots
for r in results:
    clf = r['model']
    name = r['name']
    try:
        # prefer predict_proba, fall back to decision_function
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test_scaled)
        else:
            print(f"Skipping ROC for {name}: no score method available")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_roc.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

        # store curve for combined plot
        r['roc_curve'] = (fpr, tpr, roc_auc)
    except Exception as e:
        print(f"Failed to compute/save ROC for {name}: {e}")

# combined ROC plot
plt.figure(figsize=(8, 6))
for r in results:
    if r.get('roc_curve') is not None:
        fpr, tpr, roc_auc = r['roc_curve']
        plt.plot(fpr, tpr, lw=2, label=f"{r['name']} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.6)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc='lower right')
combined_out = os.path.join(OUTPUT_DIR, 'combined_roc.png')
plt.savefig(combined_out, bbox_inches='tight')
plt.close()


# Cross-validation for SVM and Random Forest
print("\n=== Cross-validation (5-fold Stratified) for SVM and Random Forest ===")
scoring = {
    'accuracy': 'accuracy',
    'kappa': make_scorer(cohen_kappa_score),
    'roc_auc': 'roc_auc'
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

# Use pipelines so scaling is done inside each CV fold
svm_cv_pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=1234))
rf_cv_pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=1234))

for name, pipe in [("SVM (RBF)", svm_cv_pipe), ("Random Forest", rf_cv_pipe)]:
    try:
        res_cv = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        print(f"\n{name} CV results (5-fold):")
        for metric in scoring.keys():
            vals = res_cv.get(f"test_{metric}")
            if vals is not None:
                print(f"  {metric}: mean={vals.mean():.4f}, std={vals.std():.4f}")
    except Exception as e:
        print(f"Failed to run CV for {name}: {e}")