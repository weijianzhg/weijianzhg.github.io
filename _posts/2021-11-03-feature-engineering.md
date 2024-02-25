---
layout: post
title: Feature Engineering in a Pipeline
date: 2021-11-03
description: Build a pipeline to do feature engineering with Scikit-learn.
tags: feature-engineering, pipeline, scikit-learn
categories: machine-learning-engineering
giscus_comments: false
related_posts: false
thumbnail: assets/img/feature-engineering-one.png
toc:
  beginning: true
---

## Introduction

Feature engineering is a process of transforming the given dataset into a form which is easy
for the machine learning model to interpret. If we have different transformation functions for
training and prediction we may duplicate the same work and it's harder to maintain
(make some changes in one pipeline means we have to update the other pipeline as well).

{% include figure.liquid loading="eager" path="assets/img/feature-engineering-separate.png" class="img-fluid rounded z-depth-1" width="360px" zoomable=true %}

One common practice in producitionzing machine learning models is to write a transformation
_pipeline_ so that we can use the same data transformation code for both training and prediction.

{% include figure.liquid loading="eager" path="assets/img/feature-engineering-one.png" class="img-fluid rounded z-depth-1" width="360px" zoomable=true %}

In this article, we discuss how we can use [scikit-learn](https://scikit-learn.org/stable/) to build a feature engineering pipeline. Let's first have a look at a few common transformations for numeric features and categorical features.

## Transforming Numerical Features

One thing I really like about scikit-learn is that I can use the same ''fit'' and ''predict''
pattern for data preprocessing. For a preprocessor, the two methods are called `fit` and `transform`.

{% include figure.liquid loading="eager" path="assets/img/feature-engineering-preprocessor.png" class="img-fluid rounded z-depth-1" width="360px" zoomable=true %}

We can use `SimpleImputer` to complete missing values and `StandardScaler` to standardize values by
removing the mean and scaling to unit variance.

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
```

Let's create a simple example.

```python
data = {'n1': [20, 300, 400, None, 100],
      'n2': [0.1, None, 0.5, 0.6, None],
      'n3': [-20, -10, 0, -30, None],
    }

df = pd.DataFrame(data)
```

```python
df
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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.0</td>
      <td>0.1</td>
      <td>-20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300.0</td>
      <td>NaN</td>
      <td>-10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>400.0</td>
      <td>0.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>0.6</td>
      <td>-30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

We can have a look the mean of each column using the `.mean()` method.

```python
df.mean()
```

    n1    205.0
    n2      0.4
    n3    -15.0
    dtype: float64

Here we create a `SimpleImputer` object with `strategy="mean"`. This means we fill the missing value using the mean along each column.

```python
num_imputer = SimpleImputer(strategy="mean")
```

We first fit our imputer `num_imputer` on our simple dataset.

```python
num_imputer.fit(df)
```

    SimpleImputer()

After fitting the model, the statistic, i.e., the fill value for each column, is _stored_ within the imputer `num_imputer`.

```python
num_imputer.statistics_
```

    array([205. ,   0.4, -15. ])

Now we can fill the missing values in our original dataset with the `transform` method.
By the way, we can also apply fit and transform in one go with the `fit_transform` method.

```python
imputed_features = num_imputer.transform(df)
```

```python
imputed_features
```

    array([[ 2.00e+01,  1.00e-01, -2.00e+01],
           [ 3.00e+02,  4.00e-01, -1.00e+01],
           [ 4.00e+02,  5.00e-01,  0.00e+00],
           [ 2.05e+02,  6.00e-01, -3.00e+01],
           [ 1.00e+02,  4.00e-01, -1.50e+01]])

```python
type(imputed_features)
```

    numpy.ndarray

The transformed features are stored as `numpy.ndarray`. We can convert it back to `pandas.DataFrame` with

```python
imputed_df = pd.DataFrame(imputed_features,
    index=df.index, columns=df.columns)
```

```python
imputed_df
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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.0</td>
      <td>0.1</td>
      <td>-20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300.0</td>
      <td>0.4</td>
      <td>-10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>400.0</td>
      <td>0.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>205.0</td>
      <td>0.6</td>
      <td>-30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.0</td>
      <td>0.4</td>
      <td>-15.0</td>
    </tr>
  </tbody>
</table>
</div>

The cool thing is that now we can use the same statistic saved in `num_imputer` to transform other datasets. For example here we create a new dataset with only one row.

```python
# New data

data_new = {'n1': [None],
      'n2': [0.1],
      'n3': [None],
    }

df_new = pd.DataFrame(data_new)
```

```python
df_new
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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>0.1</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

We can apply `num_imputer.transform` on this new dataset to fill the missing values.

```python
pd.DataFrame(num_imputer.transform(df_new),
    index=df_new.index, columns=df_new.columns)
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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>205.0</td>
      <td>0.1</td>
      <td>-15.0</td>
    </tr>
  </tbody>
</table>
</div>

`StandardScaler` works in a similar way. Here we scale the dataset after the imputer step.

```python
num_scaler = StandardScaler()
```

```python
num_scaler.fit(imputed_df)
```

    StandardScaler()

```python
pd.DataFrame(num_scaler.transform(imputed_df),
    index=df.index, columns=df.columns)
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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.361620</td>
      <td>-1.792843e+00</td>
      <td>-0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.699210</td>
      <td>-3.317426e-16</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.435221</td>
      <td>5.976143e-01</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>1.195229e+00</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.772811</td>
      <td>-3.317426e-16</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

## Transforming Categorical Features

`OneHotEncoder` is commonly used to transform categorical features. Essentially, for each unique value in the original categorical column, a new column is created to represent this value. Each column is filled up with zeros (the value exists)
and ones (the value doesn't exist).

```python
from sklearn.preprocessing import OneHotEncoder


cat_encoder = OneHotEncoder(handle_unknown='ignore')

data = {'c1': ['Male', 'Female', 'Male', 'Female', 'Female'],
      'c2': ['Apple', 'Orange', 'Apple', 'Banana', 'Pear'],
    }

df = pd.DataFrame(data)

df
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>Apple</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>Orange</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>Apple</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>Banana</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>Pear</td>
    </tr>
  </tbody>
</table>
</div>

Let's first `fit` a one hot encoder to a dataset.

```python
cat_encoder.fit(df)
```

    OneHotEncoder(handle_unknown='ignore')

Note that the categories of each column is stored in attribute `.categories_`.

```python
cat_encoder.categories_
```

    [array(['Female', 'Male'], dtype=object),
     array(['Apple', 'Banana', 'Orange', 'Pear'], dtype=object)]

Here is the encoded dataset.

```python
pd.DataFrame(cat_encoder.transform(df).toarray(),
    index=df.index, columns=cat_encoder.get_feature_names_out())
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
      <th>c1_Female</th>
      <th>c1_Male</th>
      <th>c2_Apple</th>
      <th>c2_Banana</th>
      <th>c2_Orange</th>
      <th>c2_Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

We can now use `cat_encoder` to `transform` new dataset.

```python
data_new = {'c1': ['Female'], 'c2': ['Orange']}

df_new = pd.DataFrame(data_new)

df_new
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>Orange</td>
    </tr>
  </tbody>
</table>
</div>

```python
pd.DataFrame(cat_encoder.transform(df_new).toarray(),
    index=df_new.index, columns=cat_encoder.get_feature_names_out())
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
      <th>c1_Female</th>
      <th>c1_Male</th>
      <th>c2_Apple</th>
      <th>c2_Banana</th>
      <th>c2_Orange</th>
      <th>c2_Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

## Building a Feature Engineering Pipeline

### Make a Pipeline

For numerical features, we can make a `pipeline` to first fill the missing values with median and then apply standard scaler;
for categorical features, we can make a `pipeline` to first fill the missing values with the word "missing" and
then apply one hot encoder.

```python
from sklearn.pipeline import make_pipeline

numeric_transformer = make_pipeline(SimpleImputer(strategy="median"),
                                    StandardScaler())

categorical_transformer = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            OneHotEncoder(handle_unknown="ignore"),)

```

The transformer pipelines can be used the same way as the individual transformers, i.e., we can `fit` a pipeline with some data and use this pipeline to `transform` new data. For example,

```python
data = {'n1': [20, 300, 400, None, 100],
      'n2': [0.1, None, 0.5, 0.6, None],
      'n3': [-20, -10, 0, -30, None],
    }

df = pd.DataFrame(data)
df
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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.0</td>
      <td>0.1</td>
      <td>-20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300.0</td>
      <td>NaN</td>
      <td>-10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>400.0</td>
      <td>0.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>0.6</td>
      <td>-30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
numeric_transformer.fit(df)
```

    Pipeline(steps=[('simpleimputer', SimpleImputer(strategy='median')),
                    ('standardscaler', StandardScaler())])

Notice that the result is exactly the same as the example we give before (apply imputer and then scaler seperately).

```python
pd.DataFrame(numeric_transformer.transform(df), index=df.index, columns=df.columns)
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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.354113</td>
      <td>-1.950034</td>
      <td>-0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.706494</td>
      <td>0.344124</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.442425</td>
      <td>0.344124</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.029437</td>
      <td>0.917663</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.765368</td>
      <td>0.344124</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

### Compose a Column Transformer

For a real life dataset we may have both numeric features and categorical features. It would be nice to selectively
apply numeric transformation on the numeric features and categorical transformation on the categorical features.
We can accomplish this goal by composing a `ColumnTransformer`.

{% include figure.liquid loading="eager" path="assets/img/feature-engineering-selector.png" class="img-fluid rounded z-depth-1" width="660px" zoomable=true %}

The example below has columns with numeric values (`'n1'`, `'n2'`, `'n3'`) and categorical values (`'c1'`, `'c2'`).

```python
data = {'n1': [20, 300, 400, None, 100],
      'n2': [0.1, None, 0.5, 0.6, None],
      'n3': [-20, -10, 0, -30, None],
      'c1': ['Male', 'Female', None, 'Female', 'Female'],
      'c2': ['Apple', 'Orange', 'Apple', 'Banana', 'Pear'],
    }

df = pd.DataFrame(data)

df
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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.0</td>
      <td>0.1</td>
      <td>-20.0</td>
      <td>Male</td>
      <td>Apple</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300.0</td>
      <td>NaN</td>
      <td>-10.0</td>
      <td>Female</td>
      <td>Orange</td>
    </tr>
    <tr>
      <th>2</th>
      <td>400.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>None</td>
      <td>Apple</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>0.6</td>
      <td>-30.0</td>
      <td>Female</td>
      <td>Banana</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Female</td>
      <td>Pear</td>
    </tr>
  </tbody>
</table>
</div>

A `ColumnTransformer` stores a list of (name, transformer, columns) tuples as `transformers`, which allows
different columns to be transformed separately.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, ["n1", "n2", "n3"]),
                ("cat", categorical_transformer, ["c1", "c2"]),
            ]
        )
```

We `fit` all transformers on dataset `df`, transform dataset `df`, and concatenate the results with method `fit_transform`.

```python
preprocessor.fit_transform(df)
```

    array([[-1.35411306, -1.95003374, -0.5       ,  0.        ,  1.        ,
             0.        ,  1.        ,  0.        ,  0.        ,  0.        ],
           [ 0.70649377,  0.3441236 ,  0.5       ,  1.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
           [ 1.44242478,  0.3441236 ,  1.5       ,  0.        ,  0.        ,
             1.        ,  1.        ,  0.        ,  0.        ,  0.        ],
           [-0.02943724,  0.91766294, -1.5       ,  1.        ,  0.        ,
             0.        ,  0.        ,  1.        ,  0.        ,  0.        ],
           [-0.76536825,  0.3441236 ,  0.        ,  1.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  1.        ]])

After fitting the transformers, we can use `preprocessor` on new dataset.

```python
data_new = {'n1': [10],
      'n2': [None],
      'n3': [-10],
      'c1': ['Male'],
      'c2': [None],
    }

df_new = pd.DataFrame(data_new)
df_new

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
      <th>n1</th>
      <th>n2</th>
      <th>n3</th>
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>None</td>
      <td>-10</td>
      <td>Male</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

```python
preprocessor.transform(df_new)
```

    array([[-1.42770616,  0.3441236 ,  0.5       ,  0.        ,  1.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

## Design Your Own Transformers

We can design custom transformers by defining a subclass of `BaseEstimator` and `TransformerMixin`. There are three methods we need to implement: `__init__` , `fit`, and `transform`.

In the example below, we design a simple transformer to first fill missing values with zeros and divide the values by 10.

```python
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.fillna(0)
        return X/10

```

Once the custom transformer is initialized, it can be used the same way as any other transformers we discussed before. Here we use the custom transformer on column `"n3"`.

```python
custom_tansformer = CustomTransformer()
```

```python
preprocessor_custom = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, ["n1", "n2"]),
                ("custom", custom_tansformer, ["n3"]),
                ("cat", categorical_transformer, ["c1", "c2"]),
            ]
        )
```

```python
preprocessor_custom.fit_transform(df)
```

    array([[-1.35411306, -1.95003374, -2.        ,  0.        ,  1.        ,
             0.        ,  1.        ,  0.        ,  0.        ,  0.        ],
           [ 0.70649377,  0.3441236 , -1.        ,  1.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
           [ 1.44242478,  0.3441236 ,  0.        ,  0.        ,  0.        ,
             1.        ,  1.        ,  0.        ,  0.        ,  0.        ],
           [-0.02943724,  0.91766294, -3.        ,  1.        ,  0.        ,
             0.        ,  0.        ,  1.        ,  0.        ,  0.        ],
           [-0.76536825,  0.3441236 ,  0.        ,  1.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  1.        ]])

## Conclusion

In summary, we discussed how data transformation can be constructed as a pipeline. We can fit a data transformation pipeline on our training dataset and use the same pipeline to transform new dataset.
