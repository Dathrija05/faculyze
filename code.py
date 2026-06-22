import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ===================================================
# 1. LOAD & PREPARE DATA
# ===================================================

df = pd.read_csv('turkiye-student-evaluation_generic.csv')

# Encode categorical columns
for c in df.select_dtypes(include='object'):
    df[c] = LabelEncoder().fit_transform(df[c])

# Feedback columns
feedback_cols = [f"Q{i}" for i in range(1, 29)]

# Avg rating & target
df['avg_rating'] = df[feedback_cols].mean(axis=1)
df['Performance'] = pd.cut(
    df['avg_rating'],
    bins=[0, 3, 4, 5],
    labels=['Needs Improvement', 'Moderate', 'Good']
)

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['Performance'])

# Real labels
class_labels = ['Needs Improvement', 'Moderate', 'Good']

# Features
X = df[['instr', 'class', 'nb.repeat', 'attendance', 'difficulty'] + feedback_cols]


# ===================================================
# 2. FEATURE RANKING: CHI-SQUARE + DT
# ===================================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi_selector_full = SelectKBest(chi2, k=10)
chi_selector_full.fit(X_scaled, y)
chi_ranked = X.columns[chi_selector_full.scores_.argsort()[::-1]]

dt_fs = DecisionTreeClassifier(random_state=42)
dt_fs.fit(X, y)
dt_ranked = X.columns[np.argsort(dt_fs.feature_importances_)[::-1]]


# ===================================================
# 3. HYBRID LOOP + METRICS
# ===================================================

results = []

for chi_k in range(0, 11):
    dt_k = 10 - chi_k

    # Chi features
    chi_features = list(chi_ranked[:chi_k])

    # DT disjoint features
    dt_disjoint = []
    for f in dt_ranked:
        if f not in chi_features:
            dt_disjoint.append(f)
        if len(dt_disjoint) == dt_k:
            break

    hybrid = chi_features + dt_disjoint
    X_hybrid = X[hybrid]

    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    results.append([chi_k, dt_k, hybrid, acc, prec, rec, f1, cm])


# ===================================================
# 4. RESULTS TABLE
# ===================================================

results_df = pd.DataFrame(results, columns=[
    "Chi", "DT", "Features", "Accuracy",
    "Precision", "Recall", "F1", "Confusion_Matrix"
])

print("\n==================== RESULTS TABLE ====================")
print(results_df[["Chi", "DT", "Accuracy"]])


# ===================================================
# 5. BEST HYBRID RATIO
# ===================================================

best_idx = results_df['Accuracy'].idxmax()
best_row = results_df.loc[best_idx]

print("\n==================== BEST HYBRID RATIO ====================")
print(best_row)


# ======================================================
# 6. TRAIN DECISION TREE WITH BEST FEATURES
# ======================================================

X_best = df[best_row["Features"]]
y_best = y

final_dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=4,            # prevents huge trees
    min_samples_split=20,
    min_samples_leaf=10
)

# ⭐ FIT THE MODEL (IMPORTANT)
final_dt.fit(X_best, y_best)


# ======================================================
# 7. PLOT FINAL DECISION TREE
# ======================================================

plt.figure(figsize=(22, 12))
plot_tree(
    final_dt,
    feature_names=X_best.columns,
    class_names=['Needs Improvement', 'Moderate', 'Good'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.show()


# ===================================================
# 8. PRINT BEST METRICS
# ===================================================

print("\n============== FINAL BEST MODEL METRICS ==============")
print(f"Best Ratio: Chi={best_row['Chi']}, DT={best_row['DT']}")
print(f"Accuracy  : {best_row['Accuracy']:.4f}")
print(f"Precision : {best_row['Precision']:.4f}")
print(f"Recall    : {best_row['Recall']:.4f}")
print(f"F1 Score  : {best_row['F1']:.4f}")

print("\nConfusion Matrix:")
print(best_row['Confusion_Matrix'])


# ===================================================
# 9. CONFUSION MATRIX PLOT
# ===================================================

cm = best_row['Confusion_Matrix']

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_labels, yticklabels=class_labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"BEST Confusion Matrix (Chi={best_row['Chi']}, DT={best_row['DT']})")
plt.tight_layout()
plt.show()
