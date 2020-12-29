# Machine-Learning HW2



## Naive Bayes Classifier  
|Argument|Description|Default|
|---|---|---|
|-tri, --training_image|File of image training data|'data/train-images.idx3-ubyte|
|-trl, --training_label|File of label training data|'data/train-labels.idx1-ubyte'
|-tei, --testing_image|File of image testing data|'data/t10k-images.idx3-ubyte'|
|-tel, --testing_label|File of label testing data|'data/t10k-labels.idx1-ubyte'|
|-m, --mode|0: discrete mode, 1: continuous mode|0 (0-1)|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 naive_bayes.py [-tri training_image_filename] [-trl training_label_filename] [-tei testing_image_filename] [-tel testing_label_filename] [-m (0-1)] [-v (0-1)]
```



## Online Learning  
|Argument|Description|Default|
|---|---|---|
|-f, --filename|File of binary outcomes|data/testfile.txt'|
|a|Parameter a of initial beta prior|None|
|b|Parameter b of initial beta prior|None|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 online_learning.py [-f filename] <a> <b> [-v (0-1)]
```



## Prerequisites
* python >= 3.6
* numpy >= 1.19.2
