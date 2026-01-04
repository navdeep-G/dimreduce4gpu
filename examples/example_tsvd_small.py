import time

import numpy as np
from sklearn.decomposition import TruncatedSVD as SkTSVD

import dimreduce4gpu
from dimreduce4gpu import TruncatedSVD


def main() -> None:
    if not dimreduce4gpu.native_available():
        raise SystemExit("Native CUDA library not available. Build it first.")

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
    k = 2

    sk = SkTSVD(algorithm="arpack", n_components=k, random_state=42)
    t0 = time.time()
    sk.fit_transform(X)
    t1 = time.time() - t0

    gpu = TruncatedSVD(n_components=k, verbose=False)
    t2 = time.time()
    X2 = gpu.fit_transform(X)
    t3 = time.time() - t2

    print("Input shape:", X.shape)
    print("sklearn seconds:", t1)
    print("gpu seconds:", t3)
    print("gpu transformed shape:", X2.shape)


if __name__ == "__main__":
    main()
