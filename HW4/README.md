# Machine-Learning HW4



## Logistic Regression  
|Argument|Description|Default|
|---|---|---|
|N|number of data points|None|
|mx1|mean of x in D1|None|
|vx1|variance of x in D1|None|
|my1|mean of y in D1|None|
|vy1|variance of y in D1|None|
|mx2|mean x in D2|None|
|vx2|variance x in D2|None|
|my2|mean of y in D2|None|
|vy2|variance of y in D2|None|
|-m, --mode|0: logistic regression, 1: univariate gaussian data generator|0|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 logistic_regression.py <N> <mx1> <vx1> <my1> <vy1> <mx2> <vx2> <my2> <vy2> [-v (0-1)]
```  

### Univariate Gaussian Data Generator
```shell script
./$ python3 logistic_regression.py <N> <mx1> <vx1> <my1> <vy1> <mx2> <vx2> <my2> <vy2> -m 1 [-v (0-1)]
```



## EM Algorithm  
|Argument|Description|Default|
|---|---|---|
|-tri, --training_image|File of image training data|'data/train-images.idx3-ubyte'|
|-trl, --training_label|File of label training data|'data/train-labels.idx1-ubyte'|
|-tei, --testing_image|File of image testing data|'data/t10k-images.idx3-ubyte'|
|-tel, --testing_label|File of label testing data|'data/t10k-labels.idx1-ubyte'|
|-m, --mode|0: discrete mode, 1: continuous mode|0 (0-1)|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 em_algorithm.py [-tri training_image_filename] [-trl training_label_filename] [-tei testing_image_filename] [-tel testing_label_filename] [-v (0-1)]
```



## Prerequisites  
* python >= 3.6
* numpy >= 1.19.2
* matplotlib >= 3.3.2
* scipy >= 1.5.4
* numba >= 0.51.2
