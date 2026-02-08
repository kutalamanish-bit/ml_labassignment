import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)
file_path = "DCT_mal (2).csv"
data = pd.read_csv(file_path)
print("Dataset Shape:", data.shape)
print(data.head())
X = data.iloc[:, :-1]   
y = data.iloc[:, -1]    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
print("\nTrain Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nTraining Metrics")
print("Precision:", precision_score(y_train, y_train_pred, average='weighted'))
print("Recall   :", recall_score(y_train, y_train_pred, average='weighted'))
print("F1 Score :", f1_score(y_train, y_train_pred, average='weighted'))
print("\nTesting Metrics")
print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_test_pred, average='weighted'))
print("F1 Score :", f1_score(y_test, y_test_pred, average='weighted'))
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
r2 = r2_score(y_test, y_test_pred)
print("\nRegression Metrics")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2  :", r2)
np.random.seed(0)
X_small = np.random.randint(1, 11, size=(20, 2))
y_small = np.array([0 if x[0] + x[1] < 12 else 1 for x in X_small])
colors = ['blue' if c == 0 else 'red' for c in y_small]
plt.scatter(X_small[:, 0], X_small[:, 1], c=colors)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Training Data (20 Points)")
plt.show()
xx, yy = np.meshgrid(np.arange(0, 10, 0.1),
                     np.arange(0, 10, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]
knn_small = KNeighborsClassifier(n_neighbors=3)
knn_small.fit(X_small, y_small)
pred_grid = knn_small.predict(grid_points)
plt.scatter(grid_points[:, 0], grid_points[:, 1],
            c=['blue' if p == 0 else 'red' for p in pred_grid],
            s=1)
plt.scatter(X_small[:, 0], X_small[:, 1],
            c=colors, edgecolors='black')
plt.title("kNN Classification (k=3)")
plt.show()
for k in [1, 3, 5, 7]:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_small, y_small)
    pred = knn_k.predict(grid_points)
    plt.figure()
    plt.scatter(grid_points[:, 0], grid_points[:, 1],
                c=['blue' if p == 0 else 'red' for p in pred],
                s=1)
    plt.scatter(X_small[:, 0], X_small[:, 1],
                c=colors, edgecolors='black')
    plt.title(f"kNN Decision Boundary (k={k})")
    plt.show()
X_proj = data.iloc[:, [0, 1]]
y_proj = y
X_tr, X_te, y_tr, y_te = train_test_split(
    X_proj, y_proj, test_size=0.3, random_state=42
)
knn_proj = KNeighborsClassifier(n_neighbors=3)
knn_proj.fit(X_tr, y_tr)
pred_proj = knn_proj.predict(X_te)
plt.scatter(X_te.iloc[:, 0], X_te.iloc[:, 1],
            c=['blue' if p == 0 else 'red' for p in pred_proj])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Project Dataset Classification")
plt.show()
param_grid = {'n_neighbors': list(range(1, 21))}
grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(X_train, y_train)
print("\nBest k value:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)