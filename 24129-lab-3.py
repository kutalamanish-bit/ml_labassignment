import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from scipy.spatial.distance import minkowski

def dot_product(A, B):
    s = 0
    for i in range(len(A)):
        s += A[i] * B[i]
    return s

def euclidean_norm(A):
    s = 0
    for i in range(len(A)):
        s += A[i] * A[i]
    return s ** 0.5

def mean_vector(X):
    n, m = X.shape
    mean = np.zeros(m)
    for j in range(m):
        s = 0
        for i in range(n):
            s += X[i, j]
        mean[j] = s / n
    return mean

def variance_vector(X):
    n, m = X.shape
    mu = mean_vector(X)
    var = np.zeros(m)
    for j in range(m):
        s = 0
        for i in range(n):
            s += (X[i, j] - mu[j]) ** 2
        var[j] = s / n
    return var

def std_vector(X):
    var = variance_vector(X)
    return np.sqrt(var)

def interclass_distance(c1, c2):
    diff = c1 - c2
    return euclidean_norm(diff)

def minkowski_distance(A, B, p):
    s = 0
    for i in range(len(A)):
        s += abs(A[i] - B[i]) ** p
    return s ** (1 / p)

def euclidean_distance(A, B):
    s = 0
    for i in range(len(A)):
        s += (A[i] - B[i]) ** 2
    return s ** 0.5

def knn_predict_one(X_train, y_train, test_vec, k):
    distances = []
    for i in range(len(X_train)):
        d = euclidean_distance(X_train[i], test_vec)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    votes = {}
    for _, label in k_nearest:
        votes[label] = votes.get(label, 0) + 1
    return max(votes, key=votes.get)

def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for i in range(len(X_test)):
        preds.append(knn_predict_one(X_train, y_train, X_test[i], k))
    return np.array(preds)

def accuracy_score(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

def confusion_matrix_binary(y_true, y_pred, positive_label):
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == positive_label and y_pred[i] == positive_label:
            TP += 1
        elif y_true[i] != positive_label and y_pred[i] != positive_label:
            TN += 1
        elif y_true[i] != positive_label and y_pred[i] == positive_label:
            FP += 1
        elif y_true[i] == positive_label and y_pred[i] != positive_label:
            FN += 1
    return TP, TN, FP, FN

def precision_recall_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1

iris = load_iris()
X = iris.data
y = iris.target

mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

print("Dataset shape:", X.shape)
print("Classes:", np.unique(y))

A = X[0]
B = X[1]

print("\nA1: Dot Product & Norm")
print("Manual dot product:", dot_product(A, B))
print("Numpy dot:", np.dot(A, B))
print("Manual norm:", euclidean_norm(A))
print("Numpy norm:", np.linalg.norm(A))

print("\nA2: Intra class spread & Inter class distance")

X0 = X[y == 0]
X1 = X[y == 1]

centroid0 = np.mean(X0, axis=0)
centroid1 = np.mean(X1, axis=0)

spread0 = np.std(X0, axis=0)
spread1 = np.std(X1, axis=0)

print("Centroid class 0:", centroid0)
print("Centroid class 1:", centroid1)
print("Spread class 0:", spread0)
print("Spread class 1:", spread1)
print("Interclass distance:", np.linalg.norm(centroid0 - centroid1))

print("\nA3: Histogram Mean & Variance")
feature_index = 0  
feat = X[:, feature_index]

mean_feat = np.mean(feat)
var_feat = np.var(feat)

print("Mean:", mean_feat)
print("Variance:", var_feat)

plt.hist(feat, bins=10)
plt.title("Histogram of Feature " + str(feature_index))
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print("\nA4: Minkowski distance p=1..10")
dist_list = []
p_values = list(range(1, 11))
for p in p_values:
    d = minkowski_distance(A, B, p)
    dist_list.append(d)

print("Distances:", dist_list)

plt.plot(p_values, dist_list, marker='o')
plt.title("Minkowski Distance vs p")
plt.xlabel("p")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

print("\nA5: Compare Minkowski manual vs scipy")
p = 3
print("Manual:", minkowski_distance(A, B, p))
print("SciPy:", minkowski(A, B, p))

print("\nA6: Train test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Train size:", len(X_train), "Test size:", len(X_test))

print("\nA7-A9: sklearn kNN k=3")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

acc = neigh.score(X_test, y_test)
print("Test accuracy:", acc)

preds = neigh.predict(X_test)
print("Predictions:", preds[:10])
print("True labels:", y_test[:10])

print("\nA10: Manual kNN k=3 compare")
manual_preds = knn_predict(X_train, y_train, X_test, k=3)
manual_acc = accuracy_score(y_test, manual_preds)

print("Manual accuracy:", manual_acc)
print("Sklearn accuracy:", acc)

print("\nA11: Accuracy for k=1..11")
k_vals = list(range(1, 12))
accs = []

for k in k_vals:
    predk = knn_predict(X_train, y_train, X_test, k)
    acck = accuracy_score(y_test, predk)
    accs.append(acck)

print("Accuracies:", accs)

plt.plot(k_vals, accs, marker='o')
plt.title("Manual kNN Accuracy vs k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

print("\nA12-A13: Confusion Matrix + metrics")
positive_label = 1
TP, TN, FP, FN = confusion_matrix_binary(y_test, manual_preds, positive_label)

print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)

precision, recall, f1 = precision_recall_f1(TP, FP, FN)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nA14: Compare kNN with matrix inversion technique (will depend on your project/model)")