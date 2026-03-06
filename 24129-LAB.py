import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
def load_data(file):
    data=pd.read_csv(file)
    return data
def split_data(X, y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test
def train_model(X_train,y_train):
    model=LinearRegression()
    model.fit(X_train,y_train)
    return model
def predict_values(model,X):
    y_pred=model.predict(X)
    return y_pred
def regression_metrics(y_true,y_pred):
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mse)
    mape=np.mean(np.abs((y_true-y_pred)/y_true))*100
    r2=r2_score(y_true,y_pred)
    return mse,rmse,mape,r2
def perform_kmeans(X,k):
    kmeans=KMeans(n_clusters=k,random_state=42,n_init="auto")
    kmeans.fit(X)
    return kmeans
def clustering_scores(X,labels):
    sil=silhouette_score(X,labels)
    ch=calinski_harabasz_score(X,labels)
    db=davies_bouldin_score(X,labels)
    return sil,ch,db
def evaluate_k_values(X):
    k_values=range(2, 10)
    sil_scores=[]
    ch_scores=[]
    db_scores=[]
    for k in k_values:
        model=perform_kmeans(X,k)
        sil,ch,db=clustering_scores(X,model.labels_)
        sil_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
    return k_values,sil_scores,ch_scores,db_scores
def elbow_method(X):
    distortions=[]
    k_range=range(2, 20)
    for k in k_range:
        kmeans=KMeans(n_clusters=k,random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    return k_range,distortions
data=load_data("D:\jaswanth\IV SEMISTER\ML\LAB\DCT_mal.csv")
X=data.drop("LABEL", axis=1)
y=data["LABEL"]
# A1: Linear Regression using one attribute
X_single=X[['0']]
X_train,X_test, y_train,y_test=split_data(X_single, y)
model=train_model(X_train,y_train)
train_pred=predict_values(model,X_train)
test_pred=predict_values(model,X_test)
train_metrics=regression_metrics(y_train,train_pred)
test_metrics=regression_metrics(y_test,test_pred)
print("Train Metrics(MSE,RMSE,MAPE,R2):",train_metrics)
print("Test Metrics(MSE,RMSE,MAPE,R2):",test_metrics)
# A3: Linear Regression using all attributes
X_train2,X_test2,y_train2,y_test2=split_data(X, y)
model2=train_model(X_train2,y_train2)
train_pred2=predict_values(model2,X_train2)
test_pred2=predict_values(model2,X_test2)
train_metrics2=regression_metrics(y_train2,train_pred2)
test_metrics2=regression_metrics(y_test2,test_pred2)
print("\nAll Feature Train Metrics:",train_metrics2)
print("All Feature Test Metrics:",test_metrics2)
# A4: KMeans Clustering (k=2)
kmeans=perform_kmeans(X_train2, 2)
sil,ch,db=clustering_scores(X_train2,kmeans.labels_)
print("\nSilhouette Score:",sil)
print("Calinski Harabasz Score:",ch)
print("Davies Bouldin Index:",db)
# A6: Evaluate different k values
k_vals,sil_scores,ch_scores,db_scores=evaluate_k_values(X_train2)
plt.plot(k_vals,sil_scores)
plt.title("Silhouette Score vs K")
plt.xlabel("K")
plt.ylabel("Score")
plt.show()
# A7: Elbow Plot
k_range, distortions=elbow_method(X_train2)
plt.plot(k_range,distortions)
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Distortion")
plt.show()