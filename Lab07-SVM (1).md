# Prediction of Breast Cancer using Support Vector Machine Model

In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a **non-probabilistic binary linear classifier** (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped ***so that the examples of the separate categories are divided by a clear gap that is as wide as possible***. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall. This gap is also called maximum margin and the SVM classifier is called ***maximum margin clasifier***.

In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
![SVM-1](./Images/SVM-1.png)

## Import libraries and load data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### Get the Data

We'll use the built in breast cancer dataset from Scikit Learn. Note the load function:


```python
from sklearn.datasets import load_breast_cancer
```


```python
cancer = load_breast_cancer()
```

**The data set is presented in a dictionary form**


```python
cancer.keys()
```

**We can grab information and arrays out of this dictionary to create data frame and understand the features**

**The description of features are as follows**


```python
print(cancer['DESCR'])
```

**Show the feature names**


```python
cancer['feature_names']
```

## Set up the DataFrame


```python
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.info()
```


```python
df.describe()
```

**Is there any missing data?**


```python
# Sum of the count of null objects in all columns of data frame
np.sum(pd.isnull(df).sum()) 
```

**What are the 'target' data in the data set?**


```python
cancer['target']
```

**Adding the target data to the DataFrame**


```python
df['Cancer'] = pd.DataFrame(cancer['target'])
df.head()
```

## Exploratory Data Analysis


### Check the relative counts of benign (0) vs malignant (1) 
### cases of cancer


```python
sns.set_style('whitegrid')
sns.countplot(x='Cancer',data=df,palette='RdBu_r')
```

**Draw boxplots of all the mean features (first 10 columns) for '0' and '1' CANCER OUTCOME**


```python
l=list(df.columns[0:10])
for i in range(len(l)-1):
    sns.boxplot(x='Cancer',y=l[i], data=df, palette='winter')
    plt.figure()
```

### Not all the features seperate out the cancer predictions equally clearly
**For example, from the following two plots it is clear that smaller <br> area generally is indicative of positive cancer detection, <br>while nothing concrete can be said from the plot of mean smoothness**


```python
f,(ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(12,6))
ax1.scatter(df['mean area'],df['Cancer'])
ax1.set_title("Cancer cases as a function of mean area", 
              fontsize=15)
ax2.scatter(df['mean smoothness'],df['Cancer'])
ax2.set_title("Cancer cases as a function of mean smoothness", 
              fontsize=15)
```

## Training and prediction

### Train Test Split


```python
# Define a dataframe with only features
df_feat = df.drop('Cancer',axis=1) 
df_feat.head()
```


```python
# Define a dataframe with only target results for 
#cancer detection
df_target = df['Cancer']
df_target.head()
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = 
train_test_split(df_feat, df_target,test_size=0.30,
                 random_state=101)
```


```python
X_train.head()
```

### Train the Support Vector Classifier


```python
from sklearn.svm import SVC
```


```python
model = SVC()
```


```python
model.fit(X_train,y_train)
```

### Predictions and Evaluations


```python
predictions = model.predict(X_test)
```


```python
from sklearn.metrics import classification_report,
confusion_matrix
```

**Notice that we are classifying everything into a single class!<br> This means our model needs to have it parameters adjusted <br> (it may also help to normalize the data)**


```python
print(confusion_matrix(y_test,predictions))
```

**As expected, the classification report card is bad**


```python
print(classification_report(y_test,predictions))
```


```python
print("Misclassification error rate:",
      round(np.mean(predictions!=y_test),3))
```

## Gridsearch

Finding the right parameters (like what C or gamma values to use) is a tricky task! But luckily, Scikit-learn has the functionality of trying a bunch of combinations and see what works best, built in with GridSearchCV! The CV stands for cross-validation.

**GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested.** 


```python
param_grid = {'C': [0.1,1, 10, 100, 1000], 
              'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
```


```python
from sklearn.model_selection import GridSearchCV
```

One of the great things about GridSearchCV is that it is a meta-estimator. It takes an estimator like SVC, and creates a new estimator, that behaves exactly the same - in this case, like a classifier. You should add refit=True and choose verbose to whatever number you want, higher the number, the more verbose (verbose just means the text output describing the process).


```python
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)
```

First, it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to built a single new model using the best parameter setting.


```python
# May take awhile!
grid.fit(X_train,y_train)
```

**You can inspect the best parameters found by GridSearchCV in the best\_params\_ attribute, and the best estimator in the best\_estimator\_ attribute**


```python
grid.best_params_
```


```python
grid.best_estimator_
```

**Then you can re-run predictions on this grid object just like you would with a normal model**


```python
grid_predictions = grid.predict(X_test)
```

**Now print the confusion matrix to see improved predictions**


```python
print(confusion_matrix(y_test,grid_predictions))
```

**Classification report shows improved F1-score**


```python
print(classification_report(y_test,grid_predictions))
```

### Another set of parameters for GridSearch


```python
param_grid = {'C': [50,75,100,125,150], 'gamma': [1e-2,1e-3,1e-4,1e-5,1e-6], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(tol=1e-5),param_grid,refit=True,verbose=1)
grid.fit(X_train,y_train)
```


```python
grid.best_estimator_
```


```python
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
```
