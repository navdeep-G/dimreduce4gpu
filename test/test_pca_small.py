from dimreduce4gpu import pca
import numpy as np
import time
from sklearn.decomposition import PCA

X = np.array([[1, 2, 3], [4, 5, 6], [7,8,9], [10,11,12]], np.float32)
#X = np.random.rand(5000000,10)
k = 2

#Exact scikit impl
pca_arpack = PCA(svd_solver="arpack", n_components=k, random_state=42)
#Randomized scikit impl
pca_random = PCA(svd_solver = "randomized", n_components=k)
#Cusolver impl
pca_cusolver = pca.PCA(n_components=k, algorithm="cusolver")
#Power impl
pca_power = pca.PCA(n_components=k, algorithm="power")

print("PCA on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
print("Original X Matrix")
print(X)

print("\n")
print("pca %s run" % pca_cusolver.algorithm)
start_time = time.time()
pca_cusolver.fit(X)
end_time = time.time() - start_time
print("Total time for pca cusolver method is " + str(end_time))
print("pca cusolver Singular Values")
print(pca_cusolver.singular_values_)
print("pca cusolver Components (V^T)")
print(pca_cusolver.components_)
print("pca cusolver Explained Variance")
print(pca_cusolver.explained_variance_)
print("pca cusvoler Explained Variance Ratio")
print(pca_cusolver.explained_variance_ratio_)

print("\n")
print("pca %s run" % pca_power.algorithm)
start_time = time.time()
pca_power.fit(X)
end_time = time.time() - start_time
print("Total time for pca power method is " + str(end_time))
print("pca power Singular Values")
print(pca_power.singular_values_)
print("pca power Components (V^T)")
print(pca_power.components_)
print("pca power Explained Variance")
print(pca_power.explained_variance_)
print("pca power Explained Variance Ratio")
print(pca_power.explained_variance_ratio_)

print("\n")
print("sklearn %s run" % pca_arpack.svd_solver)
start_sk = time.time()
pca_arpack.fit(X)
end_sk = time.time() - start_sk
print("Total time for sklearn arpack is " + str(end_sk))
print("Sklearn arpack Singular Values")
print(pca_arpack.singular_values_)
print("Sklearn arpack Components (V^T)")
print(pca_arpack.components_)
print("Sklearn arpack Explained Variance")
print(pca_arpack.explained_variance_)
print("Sklearn arpack Explained Variance Ratio")
print(pca_arpack.explained_variance_ratio_)

print("\n")
print("sklearn %s run" % pca_random.svd_solver)
start_sk = time.time()
pca_random.fit(X)
end_sk = time.time() - start_sk
print("Total time for sklearn arpack is " + str(end_sk))
print("Sklearn random Singular Values")
print(pca_random.singular_values_)
print("Sklearn random Components (V^T)")
print(pca_random.components_)
print("Sklearn random Explained Variance")
print(pca_random.explained_variance_)
print("Sklearn random Explained Variance Ratio")
print(pca_random.explained_variance_ratio_)