# Machine-Learning HW7

## Kernel Eigenfaces  
|Parameter|Description|Default|
|---|---|---|
|-i, --image|Name of the directory containing images|'data/Yale_Face_Database|
|-algo, --algorithm|Algorithm to be used|0 (0: PCA, 1: LDA)|
|-m, --mode|Mode for PCA/LDA|0 (0: simple, 1: kernel)|
|-k, --k_neighbors|Number of nearest neighbors to decide classification|5|
|-ker, --kernel|Kernel type|0 (0 for linear, 1 for RBF)|
|-g, --gamma|Gamma of RBF|0.000001|
|-v, --verbosity|verbosity level|0 (0-1)|
```shell
$ python3 kernel_eigenfaces.py [-i name_of_directory] [-algo (0-1)] [-m (0-1)] [-k neighbors] [-ker (0-1)] [-g gamma] [-v (0-1)]
```

## t-SNE  
Original t-SNE code is from [t-SNE](https://lvdmaaten.github.io/tsne/).  

## Prerequisites
* numpy >= 1.19.2
* scipy >= 1.5.4
* Pillow >= 7.2.0
* matplotlib >= 3.3.2