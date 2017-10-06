from tsvd import truncated_svd
import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip

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
trunc = truncated_svd.TruncatedSVD(n_components=k) #Not really using k yet...
trunc.fit(X)
end_time = time.time() - start_time
print("Total time for tsvd is " + str(end_time))
print("Q matrix (V^T)")
print(trunc.components_)
print("w matrix (sigma/singular values)")
print(trunc.singular_values_)
print("U matrix")
print(trunc.U)
print("Original X Matrix")
print(trunc.X)
print("U * Sigma tsvd")
x_tsvd_transformed = trunc.U * trunc.singular_values_
print(x_tsvd_transformed)
print(np.var(x_tsvd_transformed, axis=0))

U, Sigma, VT = svds(X, k=2, tol=0)
# svds doesn't abide by scipy.linalg.svd/randomized_svd
# conventions, so reverse its outputs.
Sigma = Sigma[::-1]
U, VT = svd_flip(U[:, ::-1], VT[::-1])
print("sklearn U")
print(U)
print("Sklearn vt")
print(VT)
print("Sklearn sigma")
print(Sigma)
print("U * Sigma")
X_transformed = U * Sigma
print(X_transformed)
print(np.var(X_transformed, axis=0))


print("Sklearn")
start_sk = time.time()
svd.fit(X)
end_sk = time.time() - start_sk
print("Total time for sklearn is " + str(end_sk))
print("Singular Values")
print(svd.singular_values_)
print("Components (V^T)")
print(svd.components_)
print("Explained Var")
print(svd.explained_variance_)
