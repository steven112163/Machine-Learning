# Machine-Learning HW3



## Sequential Estimator  
|Argument|Description|Default|
|---|---|---|
|m|expectation value or mean|0.0|
|s|variance|1.0|
|-m, --mode|0: sequential estimator, 1: univariate gaussian data generator|0 (0-1)|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 sequential_estimator.py <m> <s> [-v (0-1)]
```  

### Univariate Gaussian Data Generator  
```shell script
./$ python3 sequential_estimator.py <m> <s> -m 1 [-v (0-1)]
```



## Bayesian Linear Regression  
|Argument|Description|Default|
|---|---|---|
|n|basis number|1|
|a|variance|1.0|
|m|weight|None|
|b|precision|1.0|
|-m, --mode|0: Bayesian Linear regression, 1: polynomial basis linear model data generator|0 (0-1)|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 bayesian_linear_regression.py <n> <a> <omega> <b> [-v (0-1)]
```  

### Polynomial Basis Linear Model Data Generator  
```shell script
./$ python3 bayesian_linear_regression.py <n> <a> <omega> <b> -m 1 [-v (0-1)]
```



## Prerequisites  
* python >= 3.6
* numpy >= 1.19.2
* matplotlib >= 3.3.2
