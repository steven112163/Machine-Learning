# Machine-Learning HW5



## Gaussian Process
|Argument|Description|Default|
|---|---|---|
|-d, --data|File of training data|'data/input.data'|
|-n, --noise|noise of the function generating data|5.0|
|-m, --mode|0: gaussian process without optimization, 1: gaussian process with optimization|0 (0-1)|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 gaussian_process.py [-d input.data] [-n noise] [-m (0-1)] [-v (0-1)]
```



## SVM
|Argument|Description|Default|
|---|---|---|
|-tri, --training_image|File of image training data|'data/X_train.csv'|
|-trl, --training_label|File of label training data|'data/Y_train.csv'|
|-tei, --testing_image|File of image testing data|'data/X_test.csv'|
|-tel, --testing_label|File of label testing data|'data/Y_test.csv'|
|-m, --mode|0: linear, polynomial and RBF comparison. 1: soft-margin SVM. 2: linear+RBF, linear, polynomial and RBF comparison|0 (0-2)|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 svm.py [-tri training_image_filename] [-trl training_label_filename] [-tei testing_image_filename] [-tel testing_label_filename] [-m (0-2)] [-v (0-1)]
```



## Prerequisites
* python >= 3.6
* numpy >= 1.19.2
* matplotlib >= 3.3.2
* scipy >= 1.5.4
* libsvm >= 3.23.0.4