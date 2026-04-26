import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import shap
import lime
import lime.lime_tabular


# =========================
# 1. LOAD DATA
# =========================
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Keep only numeric columns (important)
    df = df.select_dtypes(include=[np.number])

    return df


# =========================
# 2. SPLIT DATA
# =========================
def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 3. HEATMAP
# =========================
def plot_heatmap(df):
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()


# =========================
# 4. SCALING
# =========================
def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# =========================
# 5. PCA
# =========================
def apply_pca(X_train, X_test, variance):
    pca = PCA(n_components=variance)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    return X_train_pca, X_test_pca, pca, cumulative_variance


# =========================
# 6. MODEL TRAINING
# =========================
def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(
        max_iter=5000,
        solver='saga',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, acc, report


# =========================
# 7. SFS
# =========================
def apply_sfs(X_train, y_train, n_features):
    model = LogisticRegression(max_iter=5000, solver='saga')

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction="forward"
    )

    sfs.fit(X_train, y_train)

    return sfs


# =========================
# 8. LIME
# =========================
def lime_explain(model, X_train, X_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=None,
        class_names=[str(c) for c in np.unique(model.predict(X_train))],
        mode='classification'
    )

    exp = explainer.explain_instance(
        X_test[0],
        model.predict_proba
    )

    return exp


# =========================
# 9. SHAP
# =========================
def shap_explain(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    return shap_values


# =========================
# MAIN PROGRAM
# =========================
if __name__ == "__main__":

    # Load dataset
    df = load_data("DCT_mal.csv")
    target_column = df.columns[-1]

    # Heatmap
    plot_heatmap(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # Scaling (IMPORTANT)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # ================= PCA 99% =================
    X_train_pca_99, X_test_pca_99, pca_99, var_99 = apply_pca(
        X_train_scaled, X_test_scaled, 0.99
    )

    model_99, acc_99, rep_99 = train_model(
        X_train_pca_99, X_test_pca_99, y_train, y_test
    )

    # ================= PCA 95% =================
    X_train_pca_95, X_test_pca_95, pca_95, var_95 = apply_pca(
        X_train_scaled, X_test_scaled, 0.95
    )

    model_95, acc_95, rep_95 = train_model(
        X_train_pca_95, X_test_pca_95, y_train, y_test
    )

    # ================= SFS =================
    sfs = apply_sfs(X_train_scaled, y_train, n_features=5)

    X_train_sfs = sfs.transform(X_train_scaled)
    X_test_sfs = sfs.transform(X_test_scaled)

    model_sfs, acc_sfs, rep_sfs = train_model(
        X_train_sfs, X_test_sfs, y_train, y_test
    )

    # ================= RESULTS =================
    print("\n=== MODEL COMPARISON ===")
    print(f"PCA (99%) Accuracy: {acc_99}")
    print(f"PCA (95%) Accuracy: {acc_95}")
    print(f"SFS Accuracy: {acc_sfs}")

    # ================= LIME =================
    lime_exp = lime_explain(model_99, X_train_pca_99, X_test_pca_99)
    lime_exp.show_in_notebook()

    # ================= SHAP =================
    shap_values = shap_explain(model_99, X_train_pca_99)
    shap.summary_plot(shap_values, X_train_pca_99)