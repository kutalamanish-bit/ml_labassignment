import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("DCT_mal.csv")

data = data.dropna()

if len(data) > 300:
    data = data.sample(300, random_state=42)

data = pd.get_dummies(data)

target_col = data.columns[-1]

# -------------------- A1: BINNING --------------------
def equal_width_binning(column, bins=4):
    min_val = column.min()
    max_val = column.max()
    width = (max_val - min_val) / bins
    if width == 0:
        return pd.Series([0]*len(column))
    return ((column - min_val) / width).astype(int).clip(0, bins-1)

for col in data.columns:
    data[col] = equal_width_binning(data[col], bins=4)
# -------------------- A1: ENTROPY --------------------
def entropy(col):
    probs = col.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))
# -------------------- A2: GINI --------------------
def gini_index(col):
    probs = col.value_counts(normalize=True)
    return 1 - np.sum(probs**2)
# -------------------- A3: INFORMATION GAIN --------------------
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = data[data[feature] == v]
        weighted_entropy += (len(subset)/len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy
# -------------------- A3: BEST FEATURE --------------------
def best_feature(data, target):
    features = data.columns.drop(target)
    gains = {f: information_gain(data, f, target) for f in features}
    return max(gains, key=gains.get)
# -------------------- A5: FAST TREE --------------------
def build_tree(data, target, depth=0, max_depth=3):
    if len(data[target].unique()) == 1:
        return int(data[target].iloc[0])
    if depth >= max_depth:
        return int(data[target].mode()[0])
    best = best_feature(data, target)
    tree = {best: {}}
    for value in data[best].unique():
        subset = data[data[best] == value]
        if subset.empty:
            tree[best][value] = int(data[target].mode()[0])
        else:
            tree[best][value] = build_tree(
                subset.drop(columns=[best]),
                target,
                depth + 1,
                max_depth
            )
    return tree
# -------------------- OUTPUT --------------------
print("Entropy:", round(entropy(data[target_col]), 4))
print("Gini Index:", round(gini_index(data[target_col]), 4))
best = best_feature(data, target_col)
print("Best Root Feature:", best)
print("\nBuilding Tree...")
tree = build_tree(data, target_col)
print("\nDecision Tree:\n", tree)
# -------------------- A6: VISUALIZATION --------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, filled=True)
plt.title("Decision Tree")
plt.show()
# -------------------- A7: DECISION BOUNDARY --------------------
if X.shape[1] >= 2:
    X2 = X.iloc[:, :2]
    y2 = y
    model2 = DecisionTreeClassifier(max_depth=3)
    model2.fit(X2, y2)
    x_min, x_max = X2.iloc[:,0].min()-1, X2.iloc[:,0].max()+1
    y_min, y_max = X2.iloc[:,1].min()-1, X2.iloc[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X2.iloc[:,0], X2.iloc[:,1], c=y2)
    plt.xlabel(X2.columns[0])
    plt.ylabel(X2.columns[1])
    plt.title("Decision Boundary")
    plt.show()
else:
    print("Not enough features for decision boundary")