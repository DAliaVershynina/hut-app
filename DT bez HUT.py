import pandas as pd
from datetime import datetime
import re
import numpy as np
import warnings
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
df_transformed = pd.read_csv("data_app")
df = df_transformed
df_full = df_transformed.copy()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt

# --- Preprocessing ---
if 'Pohlavie' in df.columns:
    df['Pohlavie'] = df['Pohlavie'].map({'M': 0, 'F': 1})

blood_pressure_cols = ['A2', 'A4', 'A6', 'A8']
pulse_cols = ['A3', 'A5', 'A7', 'A9']

for col in blood_pressure_cols:
    if col in df.columns:
        df[[f'{col}_systolic', f'{col}_diastolic']] = df[col].astype(str).str.extract(r'(\d+)/(\d+)').astype(float)
        df.drop(columns=[col], inplace=True)

for col in pulse_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

block_A_cols = [
    'A1', 'A9',
    'A2_systolic', 'A2_diastolic', 'A3',
    'A4_systolic', 'A4_diastolic', 'A5',
    'A6_systolic', 'A6_diastolic', 'A7',
    'A8_systolic', 'A8_diastolic'
]
df.drop(columns=[col for col in block_A_cols if col in df.columns], inplace=True)

# --- Príprava datasetu ---
df = df.select_dtypes(include=[float, int])
df_clean = df[df["Synkopa"].isin([0, 1])].copy()
df_clean["Synkopa"] = df_clean["Synkopa"].astype(int)

X = df_clean.drop(columns=["Synkopa", "Typ Synkopy"], errors='ignore').copy()
y = df_clean["Synkopa"]
X = X.fillna(-1).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# --- Decision Tree s GridSearch ---
param_grid_dt = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid_dt,
    cv=StratifiedKFold(3),
    scoring='f1_weighted',
    n_jobs=-1
)
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

# --- Список бинарных и числовых признаков ---
binary_cols = [col for col in X.columns if set(X[col].dropna().unique()).issubset({0, 1, -1})]
numeric_cols = [col for col in X.columns if col not in binary_cols]

# --- Функция логичного отображения правил ---
def logical_export_text(decision_tree, feature_names, binary_cols):
    rules_text = export_text(decision_tree, feature_names=feature_names)
    lines = rules_text.split('\n')
    new_lines = []

    for line in lines:
        if "<=" in line or ">" in line:
            for col in binary_cols:
                if f"{col} <= 0.50" in line or f"{col} <= 0.5" in line:
                    line = line.replace(f"{col} <= 0.50", f"{col} = 0").replace(f"{col} <= 0.5", f"{col} = 0")
                elif f"{col} >  0.50" in line or f"{col} >  0.5" in line:
                    line = line.replace(f"{col} >  0.50", f"{col} = 1").replace(f"{col} >  0.5", f"{col} = 1")
        new_lines.append(line)
    return '\n'.join(new_lines)

# --- Оценка качества модели ---
y_pred = best_dt.predict(X_test)
y_proba = best_dt.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Decision Tree - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}, AUC: {roc_auc:.3f}")

# --- Визуализация дерева ---
plt.figure(figsize=(50, 40))
plot_tree(
    best_dt,
    feature_names=X.columns,
    class_names=["Negatívny", "Pozitívny"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=None
)
plt.title("Vizualizácia rozhodovacieho stromu pre predikciu synkopy (HUT test)")
plt.tight_layout()
plt.savefig("strom.png", dpi=300)

from sklearn.tree import _tree

def get_rules(tree, feature_names, binary_cols, target_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name in binary_cols:
                # Для бинарных признаков (0 или 1)
                recurse(tree_.children_left[node],
                        path + [f"({name} = 0)"], paths)
                recurse(tree_.children_right[node],
                        path + [f"({name} = 1)"], paths)
            else:
                # Для числовых признаков
                recurse(tree_.children_left[node],
                        path + [f"({name} <= {threshold:.2f})"], paths)
                recurse(tree_.children_right[node],
                        path + [f"({name} > {threshold:.2f})"], paths)
        else:
            # Лист дерева (решение)
            value = tree_.value[node][0]
            class_idx = value.argmax()
            probability = value[class_idx] / value.sum()
            path_statement = " AND ".join(path)
            rule = f"IF {path_statement} THEN class = {target_names[class_idx]} (probability = {probability:.2f})"
            paths.append(rule)

    recurse(0, path, paths)

    return paths

# Список бинарных признаков (только 0/1)
binary_cols = [col for col in X.columns if set(X[col].dropna().unique()).issubset({0, 1, -1})]

# Получение логичных правил:
rules = get_rules(best_dt, feature_names=X.columns, binary_cols=binary_cols, target_names=["Negatívny", "Pozitívny"])

# Печать первых 20 правил (для примера)
for i, rule in enumerate(rules[:100], start=1):
    print(f"R{i}: {rule}")
# --- Post-pruning pomocou cost-complexity pruning ---
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Построение пути обрезки
path = best_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Строим деревья для каждого значения ccp_alpha
dt_models = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',
        ccp_alpha=ccp_alpha
    )
    clf.fit(X_train, y_train)
    dt_models.append(clf)

# Вычисляем метрики для всех деревьев
from sklearn.metrics import f1_score

f1_scores = [f1_score(y_test, clf.predict(X_test), average='weighted') for clf in dt_models]

# Находим наилучшее дерево по F1
best_idx = np.argmax(f1_scores)
best_pruned_dt = dt_models[best_idx]

print(f"🎯 Best ccp_alpha: {ccp_alphas[best_idx]:.5f}, F1 Score: {f1_scores[best_idx]:.3f}")

# Сохранить визуализацию дерева в PNG
plt.figure(figsize=(50, 40))
plot_tree(
    best_pruned_dt,
    feature_names=X.columns,
    class_names=["Negatívny", "Pozitívny"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Orezané rozhodovacie strom (post-pruning s CCP)")
plt.tight_layout()
plt.savefig("strom.png", dpi=300)


# --- Извлечение правил из обрезанного дерева ---
rules_pruned = get_rules(
    best_pruned_dt,
    feature_names=X.columns,
    binary_cols=binary_cols,
    target_names=["Negatívny", "Pozitívny"]
)

# Печать первых 100 правил после pruning
for i, rule in enumerate(rules_pruned[:100], start=1):
    print(f"PRUNED R{i}: {rule}")
# from sklearn.linear_model import LogisticRegression
# # Обучаем модель Logistic Regression
# log_reg = LogisticRegression(max_iter=1000, random_state=42)
# log_reg.fit(X_train, y_train)
# # --- Оценка качества модели ---
# y_pred = log_reg.predict(X_test)
# y_proba = log_reg.predict_proba(X_test)[:, 1]
#
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
# recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
# f1 = f1_score(y_test, y_pred, average='weighted')
# roc_auc = roc_auc_score(y_test, y_proba)
#
# print(f"Log Reg - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}, AUC: {roc_auc:.3f}")
# # Извлекаем коэффициенты (важность признаков)
# coefficients = log_reg.coef_[0]  # Получаем коэффициенты для признаков
# selected_features = X.columns  # Признаки, которые использовались для обучения
#
# # Создаём DataFrame для важности признаков
# df_importance = pd.DataFrame({
#     "Feature": selected_features,
#     "Importance": coefficients
# }).sort_values("Importance", ascending=False)
#
# # Визуализация важности признаков
# plt.figure(figsize=(40, 40))
# plt.barh(df_importance["Feature"], df_importance["Importance"], color='purple')
# plt.xlabel("Importances")
# plt.title("Importance Logistic Regression")
# plt.gca().invert_yaxis()  # Чтобы самые важные признаки были сверху
# plt.grid(axis='x', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()
import joblib
joblib.dump(best_dt, "best_decision_tree_model.pkl")
