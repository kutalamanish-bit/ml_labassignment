import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
# A1: Linear Regression using Pseudo-Inverse
file_path = "C:\\Users\\Manu\\Desktop\\MACL\\lab Session Data.xlsx"
data=pd.read_excel(file_path, sheet_name="Purchase data")
      
print(data)

X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = data["Payment (Rs)"].values

print("Features (X):")
print(X)    
print("Payment (y):  ")
print(y)

rank_X = np.linalg.matrix_rank(X)

print("Rank of Feature Matrix:", rank_X)
X_pinv = np.linalg.pinv(X)

cost = X_pinv.dot(y)

print("Cost of Candies     :", cost[0])
print("Cost of Mangoes (Kg):", cost[1])
print("Cost of Milk Packets:", cost[2])
# A2: Classify Customers as RICH or POOR
data["Class"] = data["Payment (Rs)"].apply(
    lambda x: "RICH" if x > 200 else "POOR"
)


print(data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Class"]])

# A3: Statistical Analysis on Stock Prices
data = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")


price = data.iloc[:, 3]

mean_np = np.mean(price)
var_np = np.var(price)

print("Population Mean:", mean_np)
print("Population Variance:", var_np)

def my_mean(values):
    return sum(values) / len(values)

def my_variance(values):
    m = my_mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)

print("Mean (Own):", my_mean(price))
print("Variance (Own):", my_variance(price))

def avg_time(func, values):
    times = []
    for _ in range(10):
        start = time.time()
        func(values)
        end = time.time()
        times.append(end - start)
    return sum(times) / 10

print("Avg NumPy Time:", avg_time(np.mean, price))
print("Avg Own Time :", avg_time(my_mean, price))


wednesday_data = data[data["Day"] == "Wed"]

if len(wednesday_data) > 0:
    print("Wednesday Mean:", wednesday_data.iloc[:, 3].mean())
else:
    print("Wednesday Mean: Not Available")


april_data = data[data["Month"] == "Apr"]
print("April Mean:", april_data.iloc[:, 3].mean())


chg = data.iloc[:, 8]
loss = list(filter(lambda x: x < 0, chg))
print("Probability of Loss:", len(loss) / len(chg))

wednesday_chg = data[data["Day"] == "Wed"].iloc[:, 8]

if len(wednesday_chg) > 0:
    profit_wed = wednesday_chg[wednesday_chg > 0]
    print("Probability of Profit on Wednesday:", len(profit_wed) / len(wednesday_chg))
else:
    print("Probability of Profit on Wednesday: Not Available")


plt.scatter(data["Day"], chg)
plt.xlabel("Day")
plt.ylabel("Chg %")
plt.title("Chg% vs Day of Week")
plt.show()

# A4: Data Profiling on Thyroid Disease Dataset
thyroid = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

print("\nAttribute Datatypes:")
print(thyroid.dtypes)

categorical_cols = thyroid.select_dtypes(include=["object"]).columns
numerical_cols = thyroid.select_dtypes(exclude=["object"]).columns

print("\nCategorical Attributes:", categorical_cols)
print("Numerical Attributes:", numerical_cols)

print("\nData Range (Numerical):")
for col in numerical_cols:
    print(col, "Min:", thyroid[col].min(), "Max:", thyroid[col].max())

print("\nMissing Values:")
print(thyroid.isnull().sum())

print("\nOutlier Detection:")
for col in numerical_cols:
    Q1 = thyroid[col].quantile(0.25)
    Q3 = thyroid[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = thyroid[(thyroid[col] < Q1 - 1.5 * IQR) |
                        (thyroid[col] > Q3 + 1.5 * IQR)]
    print(col, "Outliers:", len(outliers))

print("\nMean and Variance:")
for col in numerical_cols:
    print(col, "Mean:", thyroid[col].mean(), "Variance:", thyroid[col].var())
# A5: Similarity Measures on Thyroid Disease Dataset
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

v1 = data.iloc[0]
v2 = data.iloc[1]

binary_cols = [
    col for col in data.columns
    if set(data[col].dropna().unique()).issubset({0, 1})
]

v1_bin = v1[binary_cols]
v2_bin = v2[binary_cols]

f11 = f10 = f01 = f00 = 0

for a, b in zip(v1_bin, v2_bin):
    if a == 1 and b == 1:
        f11 += 1
    elif a == 1 and b == 0:
        f10 += 1
    elif a == 0 and b == 1:
        f01 += 1
    elif a == 0 and b == 0:
        f00 += 1

if (f11 + f10 + f01) != 0:
    JC = f11 / (f11 + f10 + f01)
else:
    JC = 0


if (f11 + f10 + f01 + f00) != 0:
    SMC = (f11 + f00) / (f11 + f10 + f01 + f00)
else:
    SMC = 0

print("Jaccard Coefficient:", JC)
print("Simple Matching Coefficient:", SMC)

# A6 Cosine Similarity
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

numeric_data = data.select_dtypes(exclude=["object"])

v1 = numeric_data.iloc[0].fillna(0).values
v2 = numeric_data.iloc[1].fillna(0).values

cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Cosine Similarity:", cosine_similarity)
# A7 Jaccard Similarity Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

subset = data.iloc[:20]

binary_cols = [
    col for col in subset.columns
    if set(subset[col].dropna().unique()).issubset({0, 1})
]

bin_data = subset[binary_cols].fillna(0).values

jc_matrix = []

for i in range(len(bin_data)):
    row = []
    for j in range(len(bin_data)):
        f11 = np.sum((bin_data[i] == 1) & (bin_data[j] == 1))
        f10 = np.sum((bin_data[i] == 1) & (bin_data[j] == 0))
        f01 = np.sum((bin_data[i] == 0) & (bin_data[j] == 1))
        denom = f11 + f10 + f01
        row.append(f11 / denom if denom != 0 else 0)
    jc_matrix.append(row)

sns.heatmap(jc_matrix)
plt.title("Jaccard Similarity Heatmap")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

subset = data.iloc[:20]

binary_cols = [
    col for col in subset.columns
    if set(subset[col].dropna().unique()).issubset({0, 1})
]

bin_data = subset[binary_cols].fillna(0).values

jc_matrix = []

for i in range(len(bin_data)):
    row = []
    for j in range(len(bin_data)):
        f11 = np.sum((bin_data[i] == 1) & (bin_data[j] == 1))
        f10 = np.sum((bin_data[i] == 1) & (bin_data[j] == 0))
        f01 = np.sum((bin_data[i] == 0) & (bin_data[j] == 1))
        denom = f11 + f10 + f01
        row.append(f11 / denom if denom != 0 else 0)
    jc_matrix.append(row)

sns.heatmap(jc_matrix)
plt.title("Jaccard Similarity Heatmap")
plt.show()