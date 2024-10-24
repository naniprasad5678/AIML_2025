# K-Nearest Neighbor Classification

### Import packages and data set


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

df = pd.read_table("Classified_Data.txt",sep=',', index_col=0)
df.head()
```


```python
df.info()
```


```python
df.describe()
```

### Check the spread of the features


```python
l=list(df.columns)
l[0:len(l)-2]
```

**Run a 'for' loop to draw boxlots of all the features for '0' and '1' TARGET CLASS**


```python
sns.pairplot(df)
```

### Identify the Target Classes from the Dataset and their Counts


```python
l=list(df.columns)
l[0:len(l)-2]

for i in range(len(l)-1):
    sns.boxplot(x='TARGET CLASS',y=l[i], data=df)
    plt.figure()
```

### Scale the features using sklearn.preprocessing package

**Instantiate a scaler standardizing estimator**


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

**Fit the features data only to this estimator <br>
(leaving the TARGET CLASS column) and transform**


```python
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
```


```python
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

```

### Train/Test split, model fit and prediction


```python
from sklearn.model_selection import train_test_split
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = 
train_test_split(scaled_features,df['TARGET CLASS'],
                 test_size=0.30, random_state=101)
```


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
```


```python
pred = knn.predict(X_test)
```

**Evaluation of classification quality**


```python
from sklearn.metrics import classification_report,
confusion_matrix
conf_mat=confusion_matrix(y_test,pred)
print(conf_mat)
```


```python
print(classification_report(y_test,pred))
```


```python
print("Misclassification error rate:",round(np.mean(pred!=y_test),3))
```

**Choosing 'k' by elbow method**


```python
error_rate = []

# Will take some time
for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```


```python
plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', 
         linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value', fontsize=20)
plt.xlabel('K',fontsize=15)
plt.ylabel('Error (misclassification) Rate',fontsize=15)
```
