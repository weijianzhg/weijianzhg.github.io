---
layout: post
title: Building Custom Machine Learning Models
date: 2021-12-06
description: Building Custom Machine Learning Models
tags: machine-learning, custom-model
categories: machine-learning-engineering
giscus_comments: false
related_posts: false
thumbnail: assets/img/iris.jpeg
---


1. TOC
{:toc}


Sometimes in order to meet a specific business goal it's best to create a custom machine learning model. In this article we discuss how to create such models. We also show how use our custom machine learning models within the scikit-learn ecosystem. For example, we can apply scikit-learn's `GridSearchCV` on our custom machine learning models to find the best hyperparameters.

## Basic Components of a Machine Learning Model

In general, a (supervised) machine learning model has two main components: `fit` and `predict`.
We use the `fit` method to learn from data and use `predict` to make predictions on new data.


```python
class MLModel():

    def __init__(self):
        pass

    def fit(self, X, y):
        "train the model on a dataset"

    def predict(self, X):
        "predict y on unseen dataset"
```

## A Simple Custom Machine Learning Model For Classifying Iris Species

{% include figure.liquid path="assets/img/iris.jpeg" class="rounded float-left width: 10%" zoomable=false %}

Let's consider the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) as an example.
The dataset consists of samples from three Iris species (Iris setosa, Iris virginica, Iris versicolor)
with four features (sepal length, sepal width, petal length, petal width). We can load it from `sklearn.datasets`.


```python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True, as_frame=True)
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>




```python
X.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



It's clear that sepal length, sepal width, petal length, and petal width should be positive numbers.

Here we create a simple custom machine learning model which train the model using Support Vector Classification (SVC) but only
make predictions if all the features are positive, return `unknown` otherwise.


```python
import pandas as pd
from sklearn import svm


class MLModel():

    def __init__(self, kernel='linear', C=1.0):
        self.kernel=kernel
        self.C = C
        self.clf = svm.SVC(C=self.C, kernel=self.kernel)

    def fit(self, X: pd.DataFrame, y):
        "train the model on a dataset"
        self.clf.fit(X, y)

    def predict(self, X: pd.DataFrame):
        "predict y on unseen dataset"
        predictions = []
        for _, row in X.iterrows():
            if (row > 0).all():
                prediction = self.clf.predict(row.to_frame().T)[0]
            else:
                prediction = 'unknown'
            predictions.append(prediction)
        return predictions

model = MLModel()
```

Let's first train the model on the Iris dataset.


```python
model.fit(X, y)
```

To try our trained model we create three test samples. Note that the second sample has `0.0` sepal length and the third sample has sepal width equal to `-1.0`.


```python
X_new = pd.DataFrame({
    'sepal length (cm)': [2.3, 0, 6.3],
    'sepal width (cm)': [2.5, 3.0, -1],
    'petal length (cm)': [1.4, 4.2, 5.4],
    'petal width (cm)': [2.0, 2.3, 1.9]
})
X_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.3</td>
      <td>2.5</td>
      <td>1.4</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.2</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.3</td>
      <td>-1.0</td>
      <td>5.4</td>
      <td>1.9</td>
    </tr>
  </tbody>
</table>
</div>



As expected our model only returns a prediction for first sample and returns `unknown` for the second and the third samples.


```python
model.predict(X_new)
```




    [0, 'unknown', 'unknown']



## Using Custom Machine Learning Models within the Scikit-learn Ecosystem


In order to use our custom machine learning model within the scikit-learn ecosystem, we need to provide a few other methods:

* `get_params`: returns a dict of parameters of the machine learning model.
* `set_params`: takes a dictionary of parameters as input and sets the parameter of the machine learning model.
* `score`: provides a default evaluation criterion for the problem they are designed to solve.

We can either implement these methods ourselves or just inherit these methods from `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`.

`BaseEstimator` provides the implementation of the  `get_params` and `set_params` methods. `ClassifierMixin` provides the implementation of the `score` method as the mean accuracy.


```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class MLModel(BaseEstimator, ClassifierMixin):

    def __init__(self, kernel='linear', C=1.0):
        self.kernel=kernel
        self.C = C
        self.clf = svm.SVC(C=self.C, kernel=self.kernel)

    def fit(self, X: pd.DataFrame, y):
        "train the model on a dataset"
        self.clf.fit(X, y)

    def predict(self, X: pd.DataFrame):
        "predict y on unseen dataset"
        predictions = []
        for _, row in X.iterrows():
            if (row > 0).all():
                prediction = self.clf.predict(row.to_frame().T)[0]
            else:
                prediction = 'unknown'
            predictions.append(prediction)
        return predictions


model = MLModel()
```

Since we've defined `MLModel` as a subclass of `BaseEstimator` and `ClassifierMixin`, we can use `get_params` to retrieve all the parameters and use `score` to compute the mean accuracy on the test dataset.


```python
model.get_params()
```




    {'C': 1.0, 'kernel': 'linear'}




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model.fit(X_train, y_train)
model.score(X_test, y_test)
```




    1.0



Our custom machine learning model also works fine with scikit-learn's `GridSearchCV`.


```python
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}


clf = GridSearchCV(model, parameters)
clf.fit(X, y)
```




    GridSearchCV(estimator=MLModel(),
                 param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})




```python
clf.best_params_
```




    {'C': 1, 'kernel': 'linear'}



Note that for regression problems we need to use `RegressorMixin` instead of `ClassifierMixin`, which implements the coefficient of determination of the prediction as the `score` method. See [here](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html) for more details.
