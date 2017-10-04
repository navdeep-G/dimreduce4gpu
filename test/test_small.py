import tsvd
import numpy as np
import time
from sklearn.decomposition import TruncatedSVD

#Exact scikit impl
svd = TruncatedSVD(algorithm = "arpack", n_components=2, random_state=42, n_iter=1)

#Randomized scikit impl
#svd = TruncatedSVD(algorithm = "randomized", n_components=99, random_state=42, n_iter=5, tol=0.0)
X = np.array([[1, 2, 3], [4, 5, 6], [7,8,9], [10,11,12]], np.float32)
#X = np.random.rand(100000,100000)
k = 2

print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
print("Truncated SVD")
start_time = time.time()
trunc = tsvd.truncated_svd(X,k) #Not really using k yet...
end_time = time.time() - start_time
print("Total time for tsvd is " + str(end_time))
print(trunc[0])
print(trunc[1])
print(trunc[2])

print("Sklearn")
start_sk = time.time()
svd.fit(X)
end_sk = time.time() - start_sk
print("Total time for sklearn is " + str(end_sk))
print(svd.singular_values_)
print(svd.components_)
