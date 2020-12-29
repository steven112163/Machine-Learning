# Machine-Learning HW6



## Kernel K-means  
|Argument|Description|Default|
|---|---|---|
|-ione, --image1|First image filename|'data/image1.png'|
|-itwo, --image2|Second image filename|'data/image2.png'|
|-clu, --cluster|Number of clusters|3|
|-gs, --gammas|Parameter gamma_s in the kernel|0.0001|
|-gc, --gammac|Parameter gamma_c in the kernel|0.001|
|-m, --mode|Mode for initial clustering, 0: randomly initialized centers, 1: kmeans++|0 (0-1)|
|-v', '--verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 kernel_kmeans.py [-ione first_image_filename] [-itwo second_image_filename] [-clu number_of_clusters] [-gs gamma_s] [-gc gamma_c] [-m (0-1)] [-v (0-1)]
```



## Spectral Clustering  
|Argument|Description|Default|
|---|---|---|
|-ione, --image1|First image filename|'data/image1.png'|
|-itwo, --image2|Second image filename|'data/image2.png'|
|-clu, --cluster|Number of clusters|2|
|-gs, --gammas|Parameter gamma_s in the kernel|0.0001|
|-gc, --gammac|Parameter gamma_c in the kernel|0.001|
|-cu, --cut|Type for cut, 0: ratio cut, 1: normalized cut|0  (0-1)|
|-m, --mode|Mode for initial clustering, 0: randomly initialized centers, 1: kmeans++|0 (0-1)|
|-v, --verbosity|verbosity level|0 (0-1)|  

```shell script
./$ python3 spectral_clustering.py [-ione first_image_filename] [-itwo second_image_filename] [-clu number_of_clusters] [-gs gamma_s] [-gc gamma_c] [-cu (0-1)] [-m (0-1)] [-v (0-1)]
```



## Prerequisites  
* python >= 3.6
* numpy >= 1.19.2
* scipy >= 1.5.4
* Pillow >= 7.2.0
* matplotlib >= 3.3.2