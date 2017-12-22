from tsvd import truncated_svd
import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip

X = np.array([[1, 2, 3], [4, 5, 6], [7,8,9], [10,11,12]], np.float32)
#X = np.random.rand(5000000,10)
k = 2

#Exact scikit impl
svd_arpack = TruncatedSVD(algorithm = "arpack", n_components=k, random_state=42)
#Randomized scikit impl
svd_random = TruncatedSVD(algorithm = "randomized", n_components=k)
#Cusolver impl
trunc_cusolver = truncated_svd.TruncatedSVD(n_components=k)
#Power impl
trunc_power = truncated_svd.TruncatedSVD(n_components=k, algorithm="power")

print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
print("Original X Matrix")
print(X)

print("\n")
print("tsvd %s run" % trunc_cusolver.algorithm)
start_time = time.time()
trunc_cusolver.fit(X)
end_time = time.time() - start_time
print("Total time for tsvd cusolver method is " + str(end_time))
print("tsvd cusolver Singular Values")
print(trunc_cusolver.singular_values_)
print("tsvd cusolver Components (V^T)")
print(trunc_cusolver.components_)
print("tsvd cusolver Explained Variance")
print(trunc_cusolver.explained_variance_)
print("tsvd cusvoler Explained Variance Ratio")
print(trunc_cusolver.explained_variance_ratio_)

print("\n")
print("tsvd %s run" % trunc_power.algorithm)
start_time = time.time()
trunc_power.fit(X)
end_time = time.time() - start_time
print("Total time for tsvd power method is " + str(end_time))
print("tsvd power Singular Values")
print(trunc_power.singular_values_)
print("tsvd power Components (V^T)")
print(trunc_power.components_)
print("tsvd power Explained Variance")
print(trunc_power.explained_variance_)
print("tsvd power Explained Variance Ratio")
print(trunc_power.explained_variance_ratio_)

print("\n")
print("sklearn %s run" % svd_arpack.algorithm)
start_sk = time.time()
svd_arpack.fit(X)
end_sk = time.time() - start_sk
print("Total time for sklearn arpack is " + str(end_sk))
print("Sklearn arpack Singular Values")
print(svd_arpack.singular_values_)
print("Sklearn arpack Components (V^T)")
print(svd_arpack.components_)
print("Sklearn arpack Explained Variance")
print(svd_arpack.explained_variance_)
print("Sklearn arpack Explained Variance Ratio")
print(svd_arpack.explained_variance_ratio_)

print("\n")
print("sklearn %s run" % svd_random.algorithm)
start_sk = time.time()
svd_random.fit(X)
end_sk = time.time() - start_sk
print("Total time for sklearn arpack is " + str(end_sk))
print("Sklearn random Singular Values")
print(svd_random.singular_values_)
print("Sklearn random Components (V^T)")
print(svd_random.components_)
print("Sklearn random Explained Variance")
print(svd_random.explained_variance_)
print("Sklearn random Explained Variance Ratio")
print(svd_random.explained_variance_ratio_)

#
# print("\n")
# print("tsvd U matrix")
# print(trunc.U)
# print("tsvd V^T")
# print(trunc.components_)
# print("tsvd Sigma")
# print(trunc.singular_values_)
# print("tsvd U * Sigma")
# x_tsvd_transformed = trunc.U * trunc.singular_values_
# print(x_tsvd_transformed)
# print("tsvd Explained Variance")
# print(np.var(x_tsvd_transformed, axis=0))
#
# U, Sigma, VT = svds(X, k=2, tol=0)
# Sigma = Sigma[::-1]
# U, VT = svd_flip(U[:, ::-1], VT[::-1])
# print("\n")
# print("Sklearn U matrix")
# print(U)
# print("Sklearn V^T")
# print(VT)
# print("Sklearn Sigma")
# print(Sigma)
# print("Sklearn U * Sigma")
# X_transformed = U * Sigma
# print(X_transformed)
# print("sklearn Explained Variance")
# print(np.var(X_transformed, axis=0))
