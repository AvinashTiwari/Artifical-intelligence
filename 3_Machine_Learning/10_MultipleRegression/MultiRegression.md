

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
from sklearn.datasets import load_boston
```


```python
boston_data = load_boston()
```


```python
df = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
```


```python
df.head()
X = df
```


```python
df.shape
```




    (506, 13)




```python
y = boston_data.target
```


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
```


```python
X_constants = sm.add_constant(X)
```


```python
pd.DataFrame(X_constants)
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
      <th>const</th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.02985</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.430</td>
      <td>58.7</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.12</td>
      <td>5.21</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>0.08829</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.012</td>
      <td>66.6</td>
      <td>5.5605</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>395.60</td>
      <td>12.43</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>0.14455</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.172</td>
      <td>96.1</td>
      <td>5.9505</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>396.90</td>
      <td>19.15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>0.21124</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>5.631</td>
      <td>100.0</td>
      <td>6.0821</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>386.63</td>
      <td>29.93</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>0.17004</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.004</td>
      <td>85.9</td>
      <td>6.5921</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>386.71</td>
      <td>17.10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0.22489</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.377</td>
      <td>94.3</td>
      <td>6.3467</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>392.52</td>
      <td>20.45</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.0</td>
      <td>0.11747</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.009</td>
      <td>82.9</td>
      <td>6.2267</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>396.90</td>
      <td>13.27</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.0</td>
      <td>0.09378</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>5.889</td>
      <td>39.0</td>
      <td>5.4509</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>390.50</td>
      <td>15.71</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.0</td>
      <td>0.62976</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.949</td>
      <td>61.8</td>
      <td>4.7075</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>0.63796</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.096</td>
      <td>84.5</td>
      <td>4.4619</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>380.02</td>
      <td>10.26</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.0</td>
      <td>0.62739</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.834</td>
      <td>56.5</td>
      <td>4.4986</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>395.62</td>
      <td>8.47</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.0</td>
      <td>1.05393</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.935</td>
      <td>29.3</td>
      <td>4.4986</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>386.85</td>
      <td>6.58</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.0</td>
      <td>0.78420</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.990</td>
      <td>81.7</td>
      <td>4.2579</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>386.75</td>
      <td>14.67</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.0</td>
      <td>0.80271</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.456</td>
      <td>36.6</td>
      <td>3.7965</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>288.99</td>
      <td>11.69</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.0</td>
      <td>0.72580</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.727</td>
      <td>69.5</td>
      <td>3.7965</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>390.95</td>
      <td>11.28</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.0</td>
      <td>1.25179</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.570</td>
      <td>98.1</td>
      <td>3.7979</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>376.57</td>
      <td>21.02</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.0</td>
      <td>0.85204</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.965</td>
      <td>89.2</td>
      <td>4.0123</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>392.53</td>
      <td>13.83</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.0</td>
      <td>1.23247</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.142</td>
      <td>91.7</td>
      <td>3.9769</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>18.72</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.0</td>
      <td>0.98843</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.813</td>
      <td>100.0</td>
      <td>4.0952</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>394.54</td>
      <td>19.88</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.0</td>
      <td>0.75026</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.924</td>
      <td>94.1</td>
      <td>4.3996</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>394.33</td>
      <td>16.30</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.0</td>
      <td>0.84054</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.599</td>
      <td>85.7</td>
      <td>4.4546</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>303.42</td>
      <td>16.51</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.0</td>
      <td>0.67191</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.813</td>
      <td>90.3</td>
      <td>4.6820</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>376.88</td>
      <td>14.81</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.0</td>
      <td>0.95577</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.047</td>
      <td>88.8</td>
      <td>4.4534</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>306.38</td>
      <td>17.28</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.0</td>
      <td>0.77299</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.495</td>
      <td>94.4</td>
      <td>4.4547</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>387.94</td>
      <td>12.80</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.0</td>
      <td>1.00245</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.674</td>
      <td>87.3</td>
      <td>4.2390</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>380.23</td>
      <td>11.98</td>
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
    </tr>
    <tr>
      <th>476</th>
      <td>1.0</td>
      <td>4.87141</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.614</td>
      <td>6.484</td>
      <td>93.6</td>
      <td>2.3053</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.21</td>
      <td>18.68</td>
    </tr>
    <tr>
      <th>477</th>
      <td>1.0</td>
      <td>15.02340</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.614</td>
      <td>5.304</td>
      <td>97.3</td>
      <td>2.1007</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>349.48</td>
      <td>24.91</td>
    </tr>
    <tr>
      <th>478</th>
      <td>1.0</td>
      <td>10.23300</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.614</td>
      <td>6.185</td>
      <td>96.7</td>
      <td>2.1705</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>379.70</td>
      <td>18.03</td>
    </tr>
    <tr>
      <th>479</th>
      <td>1.0</td>
      <td>14.33370</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.614</td>
      <td>6.229</td>
      <td>88.0</td>
      <td>1.9512</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>383.32</td>
      <td>13.11</td>
    </tr>
    <tr>
      <th>480</th>
      <td>1.0</td>
      <td>5.82401</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.532</td>
      <td>6.242</td>
      <td>64.7</td>
      <td>3.4242</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>10.74</td>
    </tr>
    <tr>
      <th>481</th>
      <td>1.0</td>
      <td>5.70818</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.532</td>
      <td>6.750</td>
      <td>74.9</td>
      <td>3.3317</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>393.07</td>
      <td>7.74</td>
    </tr>
    <tr>
      <th>482</th>
      <td>1.0</td>
      <td>5.73116</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.532</td>
      <td>7.061</td>
      <td>77.0</td>
      <td>3.4106</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>395.28</td>
      <td>7.01</td>
    </tr>
    <tr>
      <th>483</th>
      <td>1.0</td>
      <td>2.81838</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.532</td>
      <td>5.762</td>
      <td>40.3</td>
      <td>4.0983</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>392.92</td>
      <td>10.42</td>
    </tr>
    <tr>
      <th>484</th>
      <td>1.0</td>
      <td>2.37857</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.583</td>
      <td>5.871</td>
      <td>41.9</td>
      <td>3.7240</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>370.73</td>
      <td>13.34</td>
    </tr>
    <tr>
      <th>485</th>
      <td>1.0</td>
      <td>3.67367</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.583</td>
      <td>6.312</td>
      <td>51.9</td>
      <td>3.9917</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>388.62</td>
      <td>10.58</td>
    </tr>
    <tr>
      <th>486</th>
      <td>1.0</td>
      <td>5.69175</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.583</td>
      <td>6.114</td>
      <td>79.8</td>
      <td>3.5459</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>392.68</td>
      <td>14.98</td>
    </tr>
    <tr>
      <th>487</th>
      <td>1.0</td>
      <td>4.83567</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.583</td>
      <td>5.905</td>
      <td>53.2</td>
      <td>3.1523</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>388.22</td>
      <td>11.45</td>
    </tr>
    <tr>
      <th>488</th>
      <td>1.0</td>
      <td>0.15086</td>
      <td>0.0</td>
      <td>27.74</td>
      <td>0.0</td>
      <td>0.609</td>
      <td>5.454</td>
      <td>92.7</td>
      <td>1.8209</td>
      <td>4.0</td>
      <td>711.0</td>
      <td>20.1</td>
      <td>395.09</td>
      <td>18.06</td>
    </tr>
    <tr>
      <th>489</th>
      <td>1.0</td>
      <td>0.18337</td>
      <td>0.0</td>
      <td>27.74</td>
      <td>0.0</td>
      <td>0.609</td>
      <td>5.414</td>
      <td>98.3</td>
      <td>1.7554</td>
      <td>4.0</td>
      <td>711.0</td>
      <td>20.1</td>
      <td>344.05</td>
      <td>23.97</td>
    </tr>
    <tr>
      <th>490</th>
      <td>1.0</td>
      <td>0.20746</td>
      <td>0.0</td>
      <td>27.74</td>
      <td>0.0</td>
      <td>0.609</td>
      <td>5.093</td>
      <td>98.0</td>
      <td>1.8226</td>
      <td>4.0</td>
      <td>711.0</td>
      <td>20.1</td>
      <td>318.43</td>
      <td>29.68</td>
    </tr>
    <tr>
      <th>491</th>
      <td>1.0</td>
      <td>0.10574</td>
      <td>0.0</td>
      <td>27.74</td>
      <td>0.0</td>
      <td>0.609</td>
      <td>5.983</td>
      <td>98.8</td>
      <td>1.8681</td>
      <td>4.0</td>
      <td>711.0</td>
      <td>20.1</td>
      <td>390.11</td>
      <td>18.07</td>
    </tr>
    <tr>
      <th>492</th>
      <td>1.0</td>
      <td>0.11132</td>
      <td>0.0</td>
      <td>27.74</td>
      <td>0.0</td>
      <td>0.609</td>
      <td>5.983</td>
      <td>83.5</td>
      <td>2.1099</td>
      <td>4.0</td>
      <td>711.0</td>
      <td>20.1</td>
      <td>396.90</td>
      <td>13.35</td>
    </tr>
    <tr>
      <th>493</th>
      <td>1.0</td>
      <td>0.17331</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0.0</td>
      <td>0.585</td>
      <td>5.707</td>
      <td>54.0</td>
      <td>2.3817</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>12.01</td>
    </tr>
    <tr>
      <th>494</th>
      <td>1.0</td>
      <td>0.27957</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0.0</td>
      <td>0.585</td>
      <td>5.926</td>
      <td>42.6</td>
      <td>2.3817</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>13.59</td>
    </tr>
    <tr>
      <th>495</th>
      <td>1.0</td>
      <td>0.17899</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0.0</td>
      <td>0.585</td>
      <td>5.670</td>
      <td>28.8</td>
      <td>2.7986</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>19.2</td>
      <td>393.29</td>
      <td>17.60</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1.0</td>
      <td>0.28960</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0.0</td>
      <td>0.585</td>
      <td>5.390</td>
      <td>72.9</td>
      <td>2.7986</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>21.14</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1.0</td>
      <td>0.26838</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0.0</td>
      <td>0.585</td>
      <td>5.794</td>
      <td>70.6</td>
      <td>2.8927</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>14.10</td>
    </tr>
    <tr>
      <th>498</th>
      <td>1.0</td>
      <td>0.23912</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0.0</td>
      <td>0.585</td>
      <td>6.019</td>
      <td>65.3</td>
      <td>2.4091</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>12.92</td>
    </tr>
    <tr>
      <th>499</th>
      <td>1.0</td>
      <td>0.17783</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0.0</td>
      <td>0.585</td>
      <td>5.569</td>
      <td>73.5</td>
      <td>2.3999</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>19.2</td>
      <td>395.77</td>
      <td>15.10</td>
    </tr>
    <tr>
      <th>500</th>
      <td>1.0</td>
      <td>0.22438</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0.0</td>
      <td>0.585</td>
      <td>6.027</td>
      <td>79.7</td>
      <td>2.4982</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>14.33</td>
    </tr>
    <tr>
      <th>501</th>
      <td>1.0</td>
      <td>0.06263</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.593</td>
      <td>69.1</td>
      <td>2.4786</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>391.99</td>
      <td>9.67</td>
    </tr>
    <tr>
      <th>502</th>
      <td>1.0</td>
      <td>0.04527</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.120</td>
      <td>76.7</td>
      <td>2.2875</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>9.08</td>
    </tr>
    <tr>
      <th>503</th>
      <td>1.0</td>
      <td>0.06076</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.976</td>
      <td>91.0</td>
      <td>2.1675</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>5.64</td>
    </tr>
    <tr>
      <th>504</th>
      <td>1.0</td>
      <td>0.10959</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.794</td>
      <td>89.3</td>
      <td>2.3889</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>393.45</td>
      <td>6.48</td>
    </tr>
    <tr>
      <th>505</th>
      <td>1.0</td>
      <td>0.04741</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.030</td>
      <td>80.8</td>
      <td>2.5050</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>7.88</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 14 columns</p>
</div>




```python
sm.OLS?
```


    [1;31mInit signature:[0m [0msm[0m[1;33m.[0m[0mOLS[0m[1;33m([0m[0mendog[0m[1;33m,[0m [0mexog[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m [0mmissing[0m[1;33m=[0m[1;34m'none'[0m[1;33m,[0m [0mhasconst[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m [1;33m**[0m[0mkwargs[0m[1;33m)[0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m     
    A simple ordinary least squares model.
    
    
    Parameters
    ----------
    endog : array-like
        1-d endogenous response variable. The dependent variable.
    exog : array-like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See
        :func:`statsmodels.tools.add_constant`.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none.'
    hasconst : None or bool
        Indicates whether the RHS includes a user-supplied constant. If True,
        a constant is not checked for and k_constant is set to 1 and all
        result statistics are calculated as if a constant is present. If
        False, a constant is not checked for and k_constant is set to 0.
    
    
    Attributes
    ----------
    weights : scalar
        Has an attribute weights = array(1.0) due to inheritance from WLS.
    
    See Also
    --------
    GLS
    
    Examples
    --------
    >>> import numpy as np
    >>>
    >>> import statsmodels.api as sm
    >>>
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>>
    >>> model = sm.OLS(Y,X)
    >>> results = model.fit()
    >>> results.params
    array([ 2.14285714,  0.25      ])
    >>> results.tvalues
    array([ 1.87867287,  0.98019606])
    >>> print(results.t_test([1, 0]))
    <T test: effect=array([ 2.14285714]), sd=array([[ 1.14062282]]), t=array([[ 1.87867287]]), p=array([[ 0.05953974]]), df_denom=5>
    >>> print(results.f_test(np.identity(2)))
    <F test: F=array([[ 19.46078431]]), p=[[ 0.00437251]], df_denom=5, df_num=2>
    
    Notes
    -----
    No constant is added by the model unless you are using formulas.
    [1;31mFile:[0m           c:\users\avinash.t\anaconda3\lib\site-packages\statsmodels\regression\linear_model.py
    [1;31mType:[0m           type
    



```python
model = sm.OLS(y, X_constants)
```


```python
lr = model.fit()
```


```python
lr.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.741</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.734</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   108.1</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 22 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>6.72e-135</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:35:47</td>     <th>  Log-Likelihood:    </th> <td> -1498.8</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3026.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   492</td>      <th>  BIC:               </th> <td>   3085.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    13</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>   36.4595</td> <td>    5.103</td> <td>    7.144</td> <td> 0.000</td> <td>   26.432</td> <td>   46.487</td>
</tr>
<tr>
  <th>CRIM</th>    <td>   -0.1080</td> <td>    0.033</td> <td>   -3.287</td> <td> 0.001</td> <td>   -0.173</td> <td>   -0.043</td>
</tr>
<tr>
  <th>ZN</th>      <td>    0.0464</td> <td>    0.014</td> <td>    3.382</td> <td> 0.001</td> <td>    0.019</td> <td>    0.073</td>
</tr>
<tr>
  <th>INDUS</th>   <td>    0.0206</td> <td>    0.061</td> <td>    0.334</td> <td> 0.738</td> <td>   -0.100</td> <td>    0.141</td>
</tr>
<tr>
  <th>CHAS</th>    <td>    2.6867</td> <td>    0.862</td> <td>    3.118</td> <td> 0.002</td> <td>    0.994</td> <td>    4.380</td>
</tr>
<tr>
  <th>NOX</th>     <td>  -17.7666</td> <td>    3.820</td> <td>   -4.651</td> <td> 0.000</td> <td>  -25.272</td> <td>  -10.262</td>
</tr>
<tr>
  <th>RM</th>      <td>    3.8099</td> <td>    0.418</td> <td>    9.116</td> <td> 0.000</td> <td>    2.989</td> <td>    4.631</td>
</tr>
<tr>
  <th>AGE</th>     <td>    0.0007</td> <td>    0.013</td> <td>    0.052</td> <td> 0.958</td> <td>   -0.025</td> <td>    0.027</td>
</tr>
<tr>
  <th>DIS</th>     <td>   -1.4756</td> <td>    0.199</td> <td>   -7.398</td> <td> 0.000</td> <td>   -1.867</td> <td>   -1.084</td>
</tr>
<tr>
  <th>RAD</th>     <td>    0.3060</td> <td>    0.066</td> <td>    4.613</td> <td> 0.000</td> <td>    0.176</td> <td>    0.436</td>
</tr>
<tr>
  <th>TAX</th>     <td>   -0.0123</td> <td>    0.004</td> <td>   -3.280</td> <td> 0.001</td> <td>   -0.020</td> <td>   -0.005</td>
</tr>
<tr>
  <th>PTRATIO</th> <td>   -0.9527</td> <td>    0.131</td> <td>   -7.283</td> <td> 0.000</td> <td>   -1.210</td> <td>   -0.696</td>
</tr>
<tr>
  <th>B</th>       <td>    0.0093</td> <td>    0.003</td> <td>    3.467</td> <td> 0.001</td> <td>    0.004</td> <td>    0.015</td>
</tr>
<tr>
  <th>LSTAT</th>   <td>   -0.5248</td> <td>    0.051</td> <td>  -10.347</td> <td> 0.000</td> <td>   -0.624</td> <td>   -0.425</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>178.041</td> <th>  Durbin-Watson:     </th> <td>   1.078</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 783.126</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 1.521</td>  <th>  Prob(JB):          </th> <td>8.84e-171</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 8.281</td>  <th>  Cond. No.          </th> <td>1.51e+04</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.51e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
form_lr = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX +  RM', data=df)
mlr = form_lr.fit()
mlr.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.587</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.582</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   118.4</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 22 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.32e-92</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:42:40</td>     <th>  Log-Likelihood:    </th> <td> -1616.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3247.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   499</td>      <th>  BIC:               </th> <td>   3276.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  -17.9546</td> <td>    3.214</td> <td>   -5.587</td> <td> 0.000</td> <td>  -24.269</td> <td>  -11.640</td>
</tr>
<tr>
  <th>CRIM</th>      <td>   -0.1769</td> <td>    0.035</td> <td>   -5.114</td> <td> 0.000</td> <td>   -0.245</td> <td>   -0.109</td>
</tr>
<tr>
  <th>ZN</th>        <td>    0.0213</td> <td>    0.014</td> <td>    1.537</td> <td> 0.125</td> <td>   -0.006</td> <td>    0.048</td>
</tr>
<tr>
  <th>INDUS</th>     <td>   -0.1437</td> <td>    0.064</td> <td>   -2.247</td> <td> 0.025</td> <td>   -0.269</td> <td>   -0.018</td>
</tr>
<tr>
  <th>CHAS</th>      <td>    4.7847</td> <td>    1.059</td> <td>    4.518</td> <td> 0.000</td> <td>    2.704</td> <td>    6.866</td>
</tr>
<tr>
  <th>NOX</th>       <td>   -7.1849</td> <td>    3.694</td> <td>   -1.945</td> <td> 0.052</td> <td>  -14.442</td> <td>    0.072</td>
</tr>
<tr>
  <th>RM</th>        <td>    7.3416</td> <td>    0.417</td> <td>   17.597</td> <td> 0.000</td> <td>    6.522</td> <td>    8.161</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>218.887</td> <th>  Durbin-Watson:     </th> <td>   0.850</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1532.877</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.738</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>10.786</td>  <th>  Cond. No.          </th> <td>    420.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python

```


```python

```


```python

```
