import scl
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import math

svd = TruncatedSVD(algorithm = "arpack", n_components=2, random_state=42, n_iter=1)
X = np.array([[1, 2, 3], [4, 5, 6], [7,8,9], [10,11,12]], np.float32)
k = 2

print("Truncated SVD")
scl.truncated_svd(X,k)
svd.fit(X)

print("Sklearn")
print(svd.singular_values_)
print(svd.components_)
