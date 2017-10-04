import tsvd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

svd = TruncatedSVD(algorithm = "arpack", n_components=2, random_state=42, n_iter=1)
X = np.array([[1, 2, 3], [4, 5, 6], [7,8,9], [10,11,12]], np.float32)
k = 2

print("Truncated SVD")
trunc = tsvd.truncated_svd(X,k)
print(trunc[0])
print(trunc[1])
print(trunc[2])

print("Sklearn")
svd.fit(X)
print(svd.singular_values_)
print(svd.components_)
