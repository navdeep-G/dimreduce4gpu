"""Simple benchmark harness.

This script is optional and will skip gracefully if the native CUDA library isn't available.
"""

import time

import numpy as np

import dimreduce4gpu


def main() -> None:
    if not dimreduce4gpu.native_available():
        print("Native CUDA library not available; skipping benchmark.")
        return

    from dimreduce4gpu import PCA

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50_000, 256)).astype(np.float32)

    pca = PCA(n_components=32, verbose=False)

    t0 = time.time()
    X2 = pca.fit_transform(X)
    t1 = time.time()

    print("Output shape:", X2.shape)
    print("Seconds:", t1 - t0)


if __name__ == "__main__":
    main()
