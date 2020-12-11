# Machine-Learning HW6

## Kernel K-means  
|Mode|Description|
|---|---|
|0|randomly initialized centers|
|1|kmeans++|
```shell script
./$ python3 kernel_kmeans.py [-ione first_image_filename] [-itwo second_image_filename] [-clu number_of_clusters] [-gs gamma_s] [-gc gamma_c] [-m (0-1)] [-v (0-1)]
```

## Spectral Clustering
```shell script
./$ python3 spectral_clustering.py [-ione first_image_filename] [-itwo second_image_filename] [-c number_of_clusters] [-v (0-1)]
```

## Prerequisites
* numpy >= 1.19.2
* scipy >= 1.5.4
* Pillow >= 7.2.0