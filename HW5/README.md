# Machine-Learning HW5

## Gaussian Process
|Mode|Description|
|---|---|
|0|without optimization|
|1|with optimization|
```shell script
./$ python3 gaussian_process.py [-d input.data] [-n noise] [-m (0-1)] [-v (0-1)]
```

## SVM  
|Mode|Description|
|---|---|
|0|linear, polynomial and RBF comparison|
|1|soft-margin SVM|
|2|linear+RBF, linear, polynomial and RBF comparison|
```shell script
./$ python3 svm.py [-tri training_image_filename] [-trl training_label_filename] [-tei testing_image_filename] [-tel testing_label_filename] [-m (0-2)] [-v (0-1)]
```

## Prerequisites
* numpy >= 1.19.2
* matplotlib >= 3.3.2
* scipy >= 1.5.4
* libsvm >= 3.23.0.4