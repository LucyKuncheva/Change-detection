# Change-detection
Change detection from streaming data

This is a collection of MATLAB methods and code for change detection in streaming data. 

* SPLL operates in the following way:

1. Two windows of data are given as input parameters. Window W1 is a matrix N1-by-n, and contains N1 objects described by n features. Window W2 is a matrix N2-by-n, containing N2 objects described by n features. The input parameter K determines how many clusters will be sought in each window (the default value is 3). 

2. The data in each window are clustered into K clusters using k-means (MATLAB Statistics Toolbox). The covariance matrices of the clusters in W1 are calculated and a weighted average covariance matrix S is calculated. The weight for a given cluster is proportional to the number of objects assigned to this cluster.

3. Each object in W2 is assigned to the cluster with the closest mean according to the Mahalanobis distance with covariance matrix S. The average of all these distances gives the first part of the SPLL criterion, SPLL1.

4. The second part, SPLL2, is calculated in the same way but windows W1 and W2 are swapped. Finally, SPLL = max(SPLL1,SPLL2).

5. A p-value is calculated using the chi-square distribution, and change is flagged if p < 0.05.

-------------- 

Source: Kuncheva L.I., Change detection in streaming multivariate data using likelihood detectors, IEEE Transactions on Knowledge and Data Engineering, 2013, 25(5), 1175-1180 (DOI: 10.1109/TKDE.2011.226)

pdf available here: http://pages.bangor.ac.uk/~mas00a/papers/lktkde13a.pdf

bibtex entry:
@ARTICLE{KunchevaTKDE13a,
author = {Ludmila I. Kuncheva},
title = {Change detection in streaming multivariate data using likelihood detectors},
journal = {{IEEE} Transactions on Knowledge and Data Engineering},
volume = {25},
number = {5},
year = {2013},
pages = {1175--1180},
doi = {10.1109/TKDE.2011.226}
}

