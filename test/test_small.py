import scl
import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
k = 3
scl.truncated_svd(X,k)
