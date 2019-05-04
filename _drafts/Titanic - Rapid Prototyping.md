
# Rapid prototyping - Titanic

## Package loading


```python
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
```


```python
%matplotlib inline
```


```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender_sub = pd.read_csv("gender_submission.csv")
```

## Basic Exploratory Data Analysis


```python
train.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
           'Fare', 'Embarked'],
          dtype='object')




```python
train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



## Rapid Prototyping

The goal with rapid prototyping is to prove that a specific project or concept is possible in the fastest, most efficient way possible. We want to answer the question:


<p style="text-align: center;"><strong>Can we get a working prototype?</strong></p>


If we can't, then we know we don't have to waste effort on an unsolvable problem. If we can solve the problem then we can work on a much deeper analysis.



For the purpose of rapid prototyping, lets impute data in the most simplest way or drop it if we need to.


```python
# Fill the age with the median value

median_age_train = train.Age.median()

# Fill missing Age, forward fill embarked, drop what we may not need for rapid prototyping
train.Age.fillna(median_age_train, inplace=True)
train.Embarked.fillna(method='ffill', inplace=True)
train.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

median_age_test = test.Age.median() # set median value

# fill NAN data
test.Age.fillna(median_age_test, inplace=True)
test.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
```

### Deep Feature Synthesis

Part of the goal of a working prototype would be to create features that can help out prototype do good work without too much work or understanding of the domain initially.

While the Titanic problem is simple enough to understand, when confronted with more difficult problems where features aren't well understood. This can be very valuable.


```python
import featuretools as ft
```


```python
full = train.append(test)
passenger_id=test['PassengerId']
```

    /anaconda3/envs/fastai-cpu/lib/python3.6/site-packages/pandas/core/frame.py:6692: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      sort=sort)



```python
# replace missing Fare
full.Fare.fillna(full.Fare.mean(), inplace=True)

# Encode Gender
full['Sex'] = full.Sex.apply(lambda x: 0 if x == "female" else 1)

# Encode Embarked
full['Embarked'] = full['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# replace all other missing with 0
full.fillna(0, inplace=True)


```


```python
# We create an entity set
es = ft.EntitySet(id = 'titanic')
```


```python
es = es.entity_from_dataframe(entity_id = 'full', dataframe = full.drop(['Survived'], axis=1), 
                              variable_types = 
                              {
                                  'Embarked': ft.variable_types.Categorical,
                                  'Sex': ft.variable_types.Boolean
                              },
                              index = 'PassengerId')

es
```




    Entityset: titanic
      Entities:
        full [Rows: 1309, Columns: 8]
      Relationships:
        No relationships




```python
es = es.normalize_entity(base_entity_id='full', new_entity_id='Embarked', index='Embarked')
es = es.normalize_entity(base_entity_id='full', new_entity_id='Sex', index='Sex')
es = es.normalize_entity(base_entity_id='full', new_entity_id='Pclass', index='Pclass')
es = es.normalize_entity(base_entity_id='full', new_entity_id='Parch', index='Parch')
es = es.normalize_entity(base_entity_id='full', new_entity_id='SibSp', index='SibSp')
es
```




    Entityset: titanic
      Entities:
        full [Rows: 1309, Columns: 8]
        Embarked [Rows: 3, Columns: 1]
        Sex [Rows: 2, Columns: 1]
        Pclass [Rows: 3, Columns: 1]
        Parch [Rows: 8, Columns: 1]
        SibSp [Rows: 7, Columns: 1]
      Relationships:
        full.Embarked -> Embarked.Embarked
        full.Sex -> Sex.Sex
        full.Pclass -> Pclass.Pclass
        full.Parch -> Parch.Parch
        full.SibSp -> SibSp.SibSp




```python
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(primitives[primitives['type'] == 'aggregation'].shape[0])
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
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num_true</td>
      <td>aggregation</td>
      <td>Counts the number of `True` values.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>std</td>
      <td>aggregation</td>
      <td>Computes the dispersion relative to the mean value, ignoring `NaN`.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sum</td>
      <td>aggregation</td>
      <td>Calculates the total addition, ignoring `NaN`.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>count</td>
      <td>aggregation</td>
      <td>Determines the total number of values, excluding `NaN`.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>num_unique</td>
      <td>aggregation</td>
      <td>Determines the number of distinct values, ignoring `NaN` values.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>skew</td>
      <td>aggregation</td>
      <td>Computes the extent to which a distribution differs from a normal distribution.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>time_since_last</td>
      <td>aggregation</td>
      <td>Calculates the time elapsed since the last datetime (in seconds).</td>
    </tr>
    <tr>
      <th>7</th>
      <td>time_since_first</td>
      <td>aggregation</td>
      <td>Calculates the time elapsed since the first datetime (in seconds).</td>
    </tr>
    <tr>
      <th>8</th>
      <td>max</td>
      <td>aggregation</td>
      <td>Calculates the highest value, ignoring `NaN` values.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>median</td>
      <td>aggregation</td>
      <td>Determines the middlemost number in a list of values.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>avg_time_between</td>
      <td>aggregation</td>
      <td>Computes the average number of seconds between consecutive events.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>all</td>
      <td>aggregation</td>
      <td>Calculates if all values are 'True' in a list.</td>
    </tr>
    <tr>
      <th>12</th>
      <td>trend</td>
      <td>aggregation</td>
      <td>Calculates the trend of a variable over time.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>min</td>
      <td>aggregation</td>
      <td>Calculates the smallest value, ignoring `NaN` values.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>any</td>
      <td>aggregation</td>
      <td>Determines if any value is 'True' in a list.</td>
    </tr>
    <tr>
      <th>15</th>
      <td>n_most_common</td>
      <td>aggregation</td>
      <td>Determines the `n` most common elements.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>percent_true</td>
      <td>aggregation</td>
      <td>Determines the percent of `True` values.</td>
    </tr>
    <tr>
      <th>17</th>
      <td>mode</td>
      <td>aggregation</td>
      <td>Determines the most commonly repeated value.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>last</td>
      <td>aggregation</td>
      <td>Determines the last value in a list.</td>
    </tr>
    <tr>
      <th>19</th>
      <td>mean</td>
      <td>aggregation</td>
      <td>Computes the average for a list of values.</td>
    </tr>
  </tbody>
</table>
</div>




```python
primitives[primitives['type'] == 'transform'].head(primitives[primitives['type'] == 'transform'].shape[0])
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
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>haversine</td>
      <td>transform</td>
      <td>Calculates the approximate haversine distance between two LatLong</td>
    </tr>
    <tr>
      <th>21</th>
      <td>multiply_numeric_scalar</td>
      <td>transform</td>
      <td>Multiply each element in the list by a scalar.</td>
    </tr>
    <tr>
      <th>22</th>
      <td>less_than_equal_to_scalar</td>
      <td>transform</td>
      <td>Determines if values are less than or equal to a given scalar.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>modulo_by_feature</td>
      <td>transform</td>
      <td>Return the modulo of a scalar by each element in the list.</td>
    </tr>
    <tr>
      <th>24</th>
      <td>num_characters</td>
      <td>transform</td>
      <td>Calculates the number of characters in a string.</td>
    </tr>
    <tr>
      <th>25</th>
      <td>time_since_previous</td>
      <td>transform</td>
      <td>Compute the time in seconds since the previous instance of an entry.</td>
    </tr>
    <tr>
      <th>26</th>
      <td>is_null</td>
      <td>transform</td>
      <td>Determines if a value is null.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>or</td>
      <td>transform</td>
      <td>Element-wise logical OR of two lists.</td>
    </tr>
    <tr>
      <th>28</th>
      <td>latitude</td>
      <td>transform</td>
      <td>Returns the first tuple value in a list of LatLong tuples.</td>
    </tr>
    <tr>
      <th>29</th>
      <td>scalar_subtract_numeric_feature</td>
      <td>transform</td>
      <td>Subtract each value in the list from a given scalar.</td>
    </tr>
    <tr>
      <th>30</th>
      <td>is_weekend</td>
      <td>transform</td>
      <td>Determines if a date falls on a weekend.</td>
    </tr>
    <tr>
      <th>31</th>
      <td>less_than_scalar</td>
      <td>transform</td>
      <td>Determines if values are less than a given scalar.</td>
    </tr>
    <tr>
      <th>32</th>
      <td>modulo_numeric</td>
      <td>transform</td>
      <td>Element-wise modulo of two lists.</td>
    </tr>
    <tr>
      <th>33</th>
      <td>not</td>
      <td>transform</td>
      <td>Negates a boolean value.</td>
    </tr>
    <tr>
      <th>34</th>
      <td>subtract_numeric</td>
      <td>transform</td>
      <td>Element-wise subtraction of two lists.</td>
    </tr>
    <tr>
      <th>35</th>
      <td>divide_numeric_scalar</td>
      <td>transform</td>
      <td>Divide each element in the list by a scalar.</td>
    </tr>
    <tr>
      <th>36</th>
      <td>greater_than_equal_to_scalar</td>
      <td>transform</td>
      <td>Determines if values are greater than or equal to a given scalar.</td>
    </tr>
    <tr>
      <th>37</th>
      <td>month</td>
      <td>transform</td>
      <td>Determines the month value of a datetime.</td>
    </tr>
    <tr>
      <th>38</th>
      <td>cum_max</td>
      <td>transform</td>
      <td>Calculates the cumulative maximum.</td>
    </tr>
    <tr>
      <th>39</th>
      <td>add_numeric</td>
      <td>transform</td>
      <td>Element-wise addition of two lists.</td>
    </tr>
    <tr>
      <th>40</th>
      <td>diff</td>
      <td>transform</td>
      <td>Compute the difference between the value in a list and the</td>
    </tr>
    <tr>
      <th>41</th>
      <td>greater_than_scalar</td>
      <td>transform</td>
      <td>Determines if values are greater than a given scalar.</td>
    </tr>
    <tr>
      <th>42</th>
      <td>minute</td>
      <td>transform</td>
      <td>Determines the minutes value of a datetime.</td>
    </tr>
    <tr>
      <th>43</th>
      <td>cum_mean</td>
      <td>transform</td>
      <td>Calculates the cumulative mean.</td>
    </tr>
    <tr>
      <th>44</th>
      <td>days_since</td>
      <td>transform</td>
      <td>Calculates the number of days from a value to a specified datetime.</td>
    </tr>
    <tr>
      <th>45</th>
      <td>not_equal</td>
      <td>transform</td>
      <td>Determines if values in one list are not equal to another list.</td>
    </tr>
    <tr>
      <th>46</th>
      <td>hour</td>
      <td>transform</td>
      <td>Determines the hour value of a datetime.</td>
    </tr>
    <tr>
      <th>47</th>
      <td>cum_sum</td>
      <td>transform</td>
      <td>Calculates the cumulative sum.</td>
    </tr>
    <tr>
      <th>48</th>
      <td>divide_numeric</td>
      <td>transform</td>
      <td>Element-wise division of two lists.</td>
    </tr>
    <tr>
      <th>49</th>
      <td>and</td>
      <td>transform</td>
      <td>Element-wise logical AND of two lists.</td>
    </tr>
    <tr>
      <th>50</th>
      <td>equal</td>
      <td>transform</td>
      <td>Determines if values in one list are equal to another list.</td>
    </tr>
    <tr>
      <th>51</th>
      <td>num_words</td>
      <td>transform</td>
      <td>Determines the number of words in a string by counting the spaces.</td>
    </tr>
    <tr>
      <th>52</th>
      <td>time_since</td>
      <td>transform</td>
      <td>Calculates time in nanoseconds from a value to a specified cutoff datetime.</td>
    </tr>
    <tr>
      <th>53</th>
      <td>longitude</td>
      <td>transform</td>
      <td>Returns the second tuple value in a list of LatLong tuples.</td>
    </tr>
    <tr>
      <th>54</th>
      <td>absolute</td>
      <td>transform</td>
      <td>Computes the absolute value of a number.</td>
    </tr>
    <tr>
      <th>55</th>
      <td>less_than_equal_to</td>
      <td>transform</td>
      <td>Determines if values in one list are less than or equal to another list.</td>
    </tr>
    <tr>
      <th>56</th>
      <td>modulo_numeric_scalar</td>
      <td>transform</td>
      <td>Return the modulo of each element in the list by a scalar.</td>
    </tr>
    <tr>
      <th>57</th>
      <td>multiply_numeric</td>
      <td>transform</td>
      <td>Element-wise multiplication of two lists.</td>
    </tr>
    <tr>
      <th>58</th>
      <td>weekday</td>
      <td>transform</td>
      <td>Determines the day of the week from a datetime.</td>
    </tr>
    <tr>
      <th>59</th>
      <td>percentile</td>
      <td>transform</td>
      <td>Determines the percentile rank for each value in a list.</td>
    </tr>
    <tr>
      <th>60</th>
      <td>subtract_numeric_scalar</td>
      <td>transform</td>
      <td>Subtract a scalar from each element in the list.</td>
    </tr>
    <tr>
      <th>61</th>
      <td>divide_by_feature</td>
      <td>transform</td>
      <td>Divide a scalar by each value in the list.</td>
    </tr>
    <tr>
      <th>62</th>
      <td>less_than</td>
      <td>transform</td>
      <td>Determines if values in one list are less than another list.</td>
    </tr>
    <tr>
      <th>63</th>
      <td>year</td>
      <td>transform</td>
      <td>Determines the year value of a datetime.</td>
    </tr>
    <tr>
      <th>64</th>
      <td>add_numeric_scalar</td>
      <td>transform</td>
      <td>Add a scalar to each value in the list.</td>
    </tr>
    <tr>
      <th>65</th>
      <td>negate</td>
      <td>transform</td>
      <td>Negates a numeric value.</td>
    </tr>
    <tr>
      <th>66</th>
      <td>greater_than_equal_to</td>
      <td>transform</td>
      <td>Determines if values in one list are greater than or equal to another list.</td>
    </tr>
    <tr>
      <th>67</th>
      <td>week</td>
      <td>transform</td>
      <td>Determines the week of the year from a datetime.</td>
    </tr>
    <tr>
      <th>68</th>
      <td>cum_min</td>
      <td>transform</td>
      <td>Calculates the cumulative minimum.</td>
    </tr>
    <tr>
      <th>69</th>
      <td>isin</td>
      <td>transform</td>
      <td>Determines whether a value is present in a provided list.</td>
    </tr>
    <tr>
      <th>70</th>
      <td>not_equal_scalar</td>
      <td>transform</td>
      <td>Determines if values in a list are not equal to a given scalar.</td>
    </tr>
    <tr>
      <th>71</th>
      <td>greater_than</td>
      <td>transform</td>
      <td>Determines if values in one list are greater than another list.</td>
    </tr>
    <tr>
      <th>72</th>
      <td>second</td>
      <td>transform</td>
      <td>Determines the seconds value of a datetime.</td>
    </tr>
    <tr>
      <th>73</th>
      <td>cum_count</td>
      <td>transform</td>
      <td>Calculates the cumulative count.</td>
    </tr>
    <tr>
      <th>74</th>
      <td>equal_scalar</td>
      <td>transform</td>
      <td>Determines if values in a list are equal to a given scalar.</td>
    </tr>
    <tr>
      <th>75</th>
      <td>day</td>
      <td>transform</td>
      <td>Determines the day of the month from a datetime.</td>
    </tr>
  </tbody>
</table>
</div>




```python
features, feature_names = ft.dfs(entityset = es, 
                                 target_entity = 'full', 
                                 max_depth = 2)
```


```python
feature_names
```




    [<Feature: Age>,
     <Feature: Fare>,
     <Feature: Parch>,
     <Feature: Pclass>,
     <Feature: SibSp>,
     <Feature: Embarked>,
     <Feature: Sex>,
     <Feature: Embarked.SUM(full.Age)>,
     <Feature: Embarked.SUM(full.Fare)>,
     <Feature: Embarked.STD(full.Age)>,
     <Feature: Embarked.STD(full.Fare)>,
     <Feature: Embarked.MAX(full.Age)>,
     <Feature: Embarked.MAX(full.Fare)>,
     <Feature: Embarked.SKEW(full.Age)>,
     <Feature: Embarked.SKEW(full.Fare)>,
     <Feature: Embarked.MIN(full.Age)>,
     <Feature: Embarked.MIN(full.Fare)>,
     <Feature: Embarked.MEAN(full.Age)>,
     <Feature: Embarked.MEAN(full.Fare)>,
     <Feature: Embarked.COUNT(full)>,
     <Feature: Embarked.NUM_UNIQUE(full.Parch)>,
     <Feature: Embarked.NUM_UNIQUE(full.Pclass)>,
     <Feature: Embarked.NUM_UNIQUE(full.SibSp)>,
     <Feature: Embarked.NUM_UNIQUE(full.Sex)>,
     <Feature: Embarked.MODE(full.Parch)>,
     <Feature: Embarked.MODE(full.Pclass)>,
     <Feature: Embarked.MODE(full.SibSp)>,
     <Feature: Embarked.MODE(full.Sex)>,
     <Feature: Sex.SUM(full.Age)>,
     <Feature: Sex.SUM(full.Fare)>,
     <Feature: Sex.STD(full.Age)>,
     <Feature: Sex.STD(full.Fare)>,
     <Feature: Sex.MAX(full.Age)>,
     <Feature: Sex.MAX(full.Fare)>,
     <Feature: Sex.SKEW(full.Age)>,
     <Feature: Sex.SKEW(full.Fare)>,
     <Feature: Sex.MIN(full.Age)>,
     <Feature: Sex.MIN(full.Fare)>,
     <Feature: Sex.MEAN(full.Age)>,
     <Feature: Sex.MEAN(full.Fare)>,
     <Feature: Sex.COUNT(full)>,
     <Feature: Sex.NUM_UNIQUE(full.Parch)>,
     <Feature: Sex.NUM_UNIQUE(full.Pclass)>,
     <Feature: Sex.NUM_UNIQUE(full.SibSp)>,
     <Feature: Sex.NUM_UNIQUE(full.Embarked)>,
     <Feature: Sex.MODE(full.Parch)>,
     <Feature: Sex.MODE(full.Pclass)>,
     <Feature: Sex.MODE(full.SibSp)>,
     <Feature: Sex.MODE(full.Embarked)>,
     <Feature: Pclass.SUM(full.Age)>,
     <Feature: Pclass.SUM(full.Fare)>,
     <Feature: Pclass.STD(full.Age)>,
     <Feature: Pclass.STD(full.Fare)>,
     <Feature: Pclass.MAX(full.Age)>,
     <Feature: Pclass.MAX(full.Fare)>,
     <Feature: Pclass.SKEW(full.Age)>,
     <Feature: Pclass.SKEW(full.Fare)>,
     <Feature: Pclass.MIN(full.Age)>,
     <Feature: Pclass.MIN(full.Fare)>,
     <Feature: Pclass.MEAN(full.Age)>,
     <Feature: Pclass.MEAN(full.Fare)>,
     <Feature: Pclass.COUNT(full)>,
     <Feature: Pclass.NUM_UNIQUE(full.Parch)>,
     <Feature: Pclass.NUM_UNIQUE(full.SibSp)>,
     <Feature: Pclass.NUM_UNIQUE(full.Embarked)>,
     <Feature: Pclass.NUM_UNIQUE(full.Sex)>,
     <Feature: Pclass.MODE(full.Parch)>,
     <Feature: Pclass.MODE(full.SibSp)>,
     <Feature: Pclass.MODE(full.Embarked)>,
     <Feature: Pclass.MODE(full.Sex)>,
     <Feature: Parch.SUM(full.Age)>,
     <Feature: Parch.SUM(full.Fare)>,
     <Feature: Parch.STD(full.Age)>,
     <Feature: Parch.STD(full.Fare)>,
     <Feature: Parch.MAX(full.Age)>,
     <Feature: Parch.MAX(full.Fare)>,
     <Feature: Parch.SKEW(full.Age)>,
     <Feature: Parch.SKEW(full.Fare)>,
     <Feature: Parch.MIN(full.Age)>,
     <Feature: Parch.MIN(full.Fare)>,
     <Feature: Parch.MEAN(full.Age)>,
     <Feature: Parch.MEAN(full.Fare)>,
     <Feature: Parch.COUNT(full)>,
     <Feature: Parch.NUM_UNIQUE(full.Pclass)>,
     <Feature: Parch.NUM_UNIQUE(full.SibSp)>,
     <Feature: Parch.NUM_UNIQUE(full.Embarked)>,
     <Feature: Parch.NUM_UNIQUE(full.Sex)>,
     <Feature: Parch.MODE(full.Pclass)>,
     <Feature: Parch.MODE(full.SibSp)>,
     <Feature: Parch.MODE(full.Embarked)>,
     <Feature: Parch.MODE(full.Sex)>,
     <Feature: SibSp.SUM(full.Age)>,
     <Feature: SibSp.SUM(full.Fare)>,
     <Feature: SibSp.STD(full.Age)>,
     <Feature: SibSp.STD(full.Fare)>,
     <Feature: SibSp.MAX(full.Age)>,
     <Feature: SibSp.MAX(full.Fare)>,
     <Feature: SibSp.SKEW(full.Age)>,
     <Feature: SibSp.SKEW(full.Fare)>,
     <Feature: SibSp.MIN(full.Age)>,
     <Feature: SibSp.MIN(full.Fare)>,
     <Feature: SibSp.MEAN(full.Age)>,
     <Feature: SibSp.MEAN(full.Fare)>,
     <Feature: SibSp.COUNT(full)>,
     <Feature: SibSp.NUM_UNIQUE(full.Parch)>,
     <Feature: SibSp.NUM_UNIQUE(full.Pclass)>,
     <Feature: SibSp.NUM_UNIQUE(full.Embarked)>,
     <Feature: SibSp.NUM_UNIQUE(full.Sex)>,
     <Feature: SibSp.MODE(full.Parch)>,
     <Feature: SibSp.MODE(full.Pclass)>,
     <Feature: SibSp.MODE(full.Embarked)>,
     <Feature: SibSp.MODE(full.Sex)>]




```python
len(feature_names)
```




    112



In a few minutes we've generated a bunch of features that we can use for prototyping our problem


```python
# Threshold for removing correlated variables
threshold = 0.95

# Absolute value correlation matrix
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head(50)
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
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Embarked</th>
      <th>Sex</th>
      <th>Embarked.SUM(full.Age)</th>
      <th>Embarked.SUM(full.Fare)</th>
      <th>Embarked.STD(full.Age)</th>
      <th>...</th>
      <th>SibSp.MEAN(full.Fare)</th>
      <th>SibSp.COUNT(full)</th>
      <th>SibSp.NUM_UNIQUE(full.Parch)</th>
      <th>SibSp.NUM_UNIQUE(full.Pclass)</th>
      <th>SibSp.NUM_UNIQUE(full.Embarked)</th>
      <th>SibSp.NUM_UNIQUE(full.Sex)</th>
      <th>SibSp.MODE(full.Parch)</th>
      <th>SibSp.MODE(full.Pclass)</th>
      <th>SibSp.MODE(full.Embarked)</th>
      <th>SibSp.MODE(full.Sex)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>NaN</td>
      <td>0.180519</td>
      <td>0.125677</td>
      <td>0.380274</td>
      <td>0.188920</td>
      <td>0.022174</td>
      <td>0.052928</td>
      <td>0.040441</td>
      <td>0.008514</td>
      <td>0.045555</td>
      <td>...</td>
      <td>6.079957e-02</td>
      <td>1.308523e-01</td>
      <td>1.976589e-01</td>
      <td>2.133987e-01</td>
      <td>2.012332e-01</td>
      <td>NaN</td>
      <td>2.379074e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.282128e-02</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.221522</td>
      <td>0.558477</td>
      <td>0.160224</td>
      <td>0.064135</td>
      <td>0.185484</td>
      <td>0.136867</td>
      <td>0.010706</td>
      <td>0.193481</td>
      <td>...</td>
      <td>2.256391e-01</td>
      <td>2.089606e-01</td>
      <td>4.979847e-02</td>
      <td>3.105973e-02</td>
      <td>9.761043e-02</td>
      <td>NaN</td>
      <td>6.134832e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.914642e-01</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.018322</td>
      <td>0.373587</td>
      <td>0.096857</td>
      <td>0.213125</td>
      <td>0.083092</td>
      <td>0.102642</td>
      <td>0.091228</td>
      <td>...</td>
      <td>3.302803e-01</td>
      <td>3.625643e-01</td>
      <td>5.262633e-02</td>
      <td>2.650650e-01</td>
      <td>2.781161e-01</td>
      <td>NaN</td>
      <td>2.938461e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.488658e-01</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.060832</td>
      <td>0.033373</td>
      <td>0.124617</td>
      <td>0.051522</td>
      <td>0.091441</td>
      <td>0.280068</td>
      <td>...</td>
      <td>9.321064e-02</td>
      <td>5.610448e-02</td>
      <td>2.076503e-01</td>
      <td>1.435907e-01</td>
      <td>1.240303e-01</td>
      <td>NaN</td>
      <td>1.488672e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.623380e-01</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.074966</td>
      <td>0.109609</td>
      <td>0.076507</td>
      <td>0.070912</td>
      <td>0.032782</td>
      <td>...</td>
      <td>7.100906e-01</td>
      <td>8.101948e-01</td>
      <td>4.109176e-01</td>
      <td>7.593949e-01</td>
      <td>7.792276e-01</td>
      <td>NaN</td>
      <td>8.217369e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.515147e-01</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.124849</td>
      <td>0.966496</td>
      <td>0.983744</td>
      <td>0.604985</td>
      <td>...</td>
      <td>7.474154e-02</td>
      <td>5.931944e-02</td>
      <td>2.727147e-02</td>
      <td>3.287548e-02</td>
      <td>8.961550e-02</td>
      <td>NaN</td>
      <td>5.740721e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.370091e-02</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.123637</td>
      <td>0.120740</td>
      <td>0.066315</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Embarked.SUM(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.904692</td>
      <td>0.380337</td>
      <td>...</td>
      <td>5.080938e-02</td>
      <td>4.266879e-02</td>
      <td>6.158505e-02</td>
      <td>5.262555e-02</td>
      <td>1.046465e-01</td>
      <td>NaN</td>
      <td>7.803594e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.143051e-02</td>
    </tr>
    <tr>
      <th>Embarked.SUM(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.738135</td>
      <td>...</td>
      <td>8.851772e-02</td>
      <td>6.861361e-02</td>
      <td>2.182977e-03</td>
      <td>1.775324e-02</td>
      <td>7.554246e-02</td>
      <td>NaN</td>
      <td>4.069646e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.454272e-02</td>
    </tr>
    <tr>
      <th>Embarked.STD(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.116884e-01</td>
      <td>8.137341e-02</td>
      <td>9.277791e-02</td>
      <td>4.479328e-02</td>
      <td>1.724522e-03</td>
      <td>NaN</td>
      <td>3.522716e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.220009e-01</td>
    </tr>
    <tr>
      <th>Embarked.STD(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>6.863038e-02</td>
      <td>4.548945e-02</td>
      <td>1.399096e-01</td>
      <td>8.583273e-02</td>
      <td>8.526836e-02</td>
      <td>NaN</td>
      <td>9.677419e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.101666e-01</td>
    </tr>
    <tr>
      <th>Embarked.MAX(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>3.953760e-02</td>
      <td>3.467741e-02</td>
      <td>7.500714e-02</td>
      <td>6.000096e-02</td>
      <td>1.089434e-01</td>
      <td>NaN</td>
      <td>8.526941e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.605402e-03</td>
    </tr>
    <tr>
      <th>Embarked.MAX(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.869541e-02</td>
      <td>5.334542e-02</td>
      <td>1.386663e-01</td>
      <td>8.311418e-02</td>
      <td>7.566791e-02</td>
      <td>NaN</td>
      <td>9.124495e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.170830e-01</td>
    </tr>
    <tr>
      <th>Embarked.SKEW(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.117975e-01</td>
      <td>8.235227e-02</td>
      <td>7.634822e-02</td>
      <td>3.321729e-02</td>
      <td>1.505507e-02</td>
      <td>NaN</td>
      <td>2.028997e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.151055e-01</td>
    </tr>
    <tr>
      <th>Embarked.SKEW(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>8.041969e-02</td>
      <td>5.470374e-02</td>
      <td>1.382240e-01</td>
      <td>8.248716e-02</td>
      <td>7.379029e-02</td>
      <td>NaN</td>
      <td>9.008990e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.181705e-01</td>
    </tr>
    <tr>
      <th>Embarked.MIN(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.042943e-01</td>
      <td>7.866970e-02</td>
      <td>3.734317e-02</td>
      <td>7.157385e-03</td>
      <td>4.846077e-02</td>
      <td>NaN</td>
      <td>1.177652e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.299433e-02</td>
    </tr>
    <tr>
      <th>Embarked.MIN(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>6.626721e-02</td>
      <td>5.348169e-02</td>
      <td>4.049116e-02</td>
      <td>4.062107e-02</td>
      <td>9.602416e-02</td>
      <td>NaN</td>
      <td>6.568088e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.181999e-02</td>
    </tr>
    <tr>
      <th>Embarked.MEAN(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.853642e-02</td>
      <td>3.771210e-02</td>
      <td>1.392973e-01</td>
      <td>8.725143e-02</td>
      <td>9.300786e-02</td>
      <td>NaN</td>
      <td>1.006344e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.024409e-01</td>
    </tr>
    <tr>
      <th>Embarked.MEAN(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>6.548557e-02</td>
      <td>4.305673e-02</td>
      <td>1.398962e-01</td>
      <td>8.639949e-02</td>
      <td>8.785982e-02</td>
      <td>NaN</td>
      <td>9.813762e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.078349e-01</td>
    </tr>
    <tr>
      <th>Embarked.COUNT(full)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4.855874e-02</td>
      <td>4.107968e-02</td>
      <td>6.438503e-02</td>
      <td>5.418260e-02</td>
      <td>1.056263e-01</td>
      <td>NaN</td>
      <td>7.958898e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.577004e-03</td>
    </tr>
    <tr>
      <th>Embarked.NUM_UNIQUE(full.Parch)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>3.634142e-02</td>
      <td>3.239726e-02</td>
      <td>7.855319e-02</td>
      <td>6.190952e-02</td>
      <td>1.098979e-01</td>
      <td>NaN</td>
      <td>8.708501e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.475036e-03</td>
    </tr>
    <tr>
      <th>Embarked.NUM_UNIQUE(full.Pclass)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Embarked.NUM_UNIQUE(full.SibSp)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2.014379e-02</td>
      <td>2.075504e-02</td>
      <td>9.492864e-02</td>
      <td>7.045971e-02</td>
      <td>1.131144e-01</td>
      <td>NaN</td>
      <td>9.484037e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.540821e-02</td>
    </tr>
    <tr>
      <th>Embarked.NUM_UNIQUE(full.Sex)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Embarked.MODE(full.Parch)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Embarked.MODE(full.Pclass)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>3.844424e-02</td>
      <td>2.245797e-02</td>
      <td>1.339125e-01</td>
      <td>8.714516e-02</td>
      <td>1.041817e-01</td>
      <td>NaN</td>
      <td>1.045429e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.529384e-02</td>
    </tr>
    <tr>
      <th>Embarked.MODE(full.SibSp)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Embarked.MODE(full.Sex)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sex.SUM(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.SUM(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.STD(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.STD(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.MAX(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.MAX(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>3.665734e-14</td>
      <td>2.195949e-16</td>
      <td>3.351113e-16</td>
      <td>1.484102e-15</td>
      <td>2.188883e-15</td>
      <td>NaN</td>
      <td>4.885604e-16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.207137e-16</td>
    </tr>
    <tr>
      <th>Sex.SKEW(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.SKEW(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.MIN(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.MIN(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.MEAN(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.MEAN(full.Fare)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.COUNT(full)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.925157e-01</td>
      <td>1.773133e-01</td>
      <td>8.506071e-02</td>
      <td>1.654746e-03</td>
      <td>4.743528e-02</td>
      <td>NaN</td>
      <td>2.062120e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.868998e-01</td>
    </tr>
    <tr>
      <th>Sex.NUM_UNIQUE(full.Parch)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sex.NUM_UNIQUE(full.Pclass)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sex.NUM_UNIQUE(full.SibSp)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sex.NUM_UNIQUE(full.Embarked)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sex.MODE(full.Parch)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sex.MODE(full.Pclass)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sex.MODE(full.SibSp)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sex.MODE(full.Embarked)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Pclass.SUM(full.Age)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.015821e-02</td>
      <td>3.925644e-02</td>
      <td>1.914864e-01</td>
      <td>1.480807e-01</td>
      <td>1.407274e-01</td>
      <td>NaN</td>
      <td>1.598170e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.314368e-01</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 112 columns</p>
</div>




```python
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d features to remove.' % (len(collinear_features)))
```

    There are 48 features to remove.



```python
features_filtered = features.drop(columns = collinear_features)

print('The number of features that passed the collinearity threshold: ', features_filtered.shape[1])
```

    The number of features that passed the collinearity threshold:  64


## Rapid XGBoost


```python
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

```


```python
features_positive = features_filtered.loc[:, features_filtered.ge(0).all()]
```


```python
features_positive
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
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Embarked</th>
      <th>Sex</th>
      <th>Embarked.STD(full.Age)</th>
      <th>Embarked.STD(full.Fare)</th>
      <th>Embarked.NUM_UNIQUE(full.Pclass)</th>
      <th>...</th>
      <th>SibSp.MEAN(full.Age)</th>
      <th>SibSp.MEAN(full.Fare)</th>
      <th>SibSp.NUM_UNIQUE(full.Parch)</th>
      <th>SibSp.NUM_UNIQUE(full.Pclass)</th>
      <th>SibSp.NUM_UNIQUE(full.Embarked)</th>
      <th>SibSp.NUM_UNIQUE(full.Sex)</th>
      <th>SibSp.MODE(full.Parch)</th>
      <th>SibSp.MODE(full.Pclass)</th>
      <th>SibSp.MODE(full.Embarked)</th>
      <th>SibSp.MODE(full.Sex)</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>28.0</td>
      <td>8.4583</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54.0</td>
      <td>51.8625</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0</td>
      <td>21.0750</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>18.650000</td>
      <td>71.332090</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>27.0</td>
      <td>11.1333</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>14.0</td>
      <td>30.0708</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4.0</td>
      <td>16.7000</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>58.0</td>
      <td>26.5500</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>39.0</td>
      <td>31.2750</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>14.0</td>
      <td>7.8542</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>55.0</td>
      <td>16.0000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.0</td>
      <td>29.1250</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>8.772727</td>
      <td>30.594318</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>28.0</td>
      <td>13.0000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>31.0</td>
      <td>18.0000</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>28.0</td>
      <td>7.2250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>35.0</td>
      <td>26.0000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>34.0</td>
      <td>13.0000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>15.0</td>
      <td>8.0292</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>28.0</td>
      <td>35.5000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>8.0</td>
      <td>21.0750</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>18.650000</td>
      <td>71.332090</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>38.0</td>
      <td>31.3875</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28.0</td>
      <td>7.2250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>19.0</td>
      <td>263.0000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>18.650000</td>
      <td>71.332090</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>28.0</td>
      <td>7.8792</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>28.0</td>
      <td>7.8958</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1280</th>
      <td>21.0</td>
      <td>7.7500</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1281</th>
      <td>6.0</td>
      <td>21.0750</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>18.650000</td>
      <td>71.332090</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1282</th>
      <td>23.0</td>
      <td>93.5000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1283</th>
      <td>51.0</td>
      <td>39.4000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1284</th>
      <td>13.0</td>
      <td>20.2500</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1285</th>
      <td>47.0</td>
      <td>10.5000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1286</th>
      <td>29.0</td>
      <td>22.0250</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>18.650000</td>
      <td>71.332090</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1287</th>
      <td>18.0</td>
      <td>60.0000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1288</th>
      <td>24.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1289</th>
      <td>48.0</td>
      <td>79.2000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1290</th>
      <td>22.0</td>
      <td>7.7750</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1291</th>
      <td>31.0</td>
      <td>7.7333</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1292</th>
      <td>30.0</td>
      <td>164.8667</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>38.0</td>
      <td>21.0000</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>22.0</td>
      <td>59.4000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1295</th>
      <td>17.0</td>
      <td>47.1000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1296</th>
      <td>43.0</td>
      <td>27.7208</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1297</th>
      <td>20.0</td>
      <td>13.8625</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>23.0</td>
      <td>10.5000</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1299</th>
      <td>50.0</td>
      <td>211.5000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1300</th>
      <td>27.0</td>
      <td>7.7208</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>3.0</td>
      <td>13.7750</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>27.0</td>
      <td>7.7500</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>37.0</td>
      <td>90.0000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>9.991200</td>
      <td>14.857148</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>28.0</td>
      <td>7.7750</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>27.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>39.0</td>
      <td>108.9000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>38.5</td>
      <td>7.2500</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1308</th>
      <td>27.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.005236</td>
      <td>37.076590</td>
      <td>3</td>
      <td>...</td>
      <td>30.168810</td>
      <td>25.793835</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>27.0</td>
      <td>22.3583</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13.632262</td>
      <td>84.036802</td>
      <td>3</td>
      <td>...</td>
      <td>30.643448</td>
      <td>48.711300</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1309 rows × 63 columns</p>
</div>




```python
train_X = features_positive[:train.shape[0]]
train_y = train['Survived']

test_X = features_positive[train.shape[0]:]
```


```python
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
```


```python
gbm = xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05, random_state=42)
gbm.fit(train_X, train_y)
cross_val_score(gbm,train_X, train_y, scoring='accuracy', cv=10).mean()
```




    0.8294841675178753




```python
gbm_pred = gbm.predict(X_test)
```


```python
print(classification_report(y_test, gbm_pred))
```

                  precision    recall  f1-score   support
    
               0       0.90      0.92      0.91       105
               1       0.89      0.85      0.87        74
    
       micro avg       0.89      0.89      0.89       179
       macro avg       0.89      0.89      0.89       179
    weighted avg       0.89      0.89      0.89       179
    



```python
gbm_pred_final = gbm.predict(test_X)
```


```python
sub = test_X.reset_index()
```


```python
sub['Survived'] = pd.DataFrame(gbm_pred_final)
```


```python
sub = sub[['PassengerId', 'Survived']]
```


```python
sub.to_csv('gbm_submission.csv', index=False)
```


```python

```
