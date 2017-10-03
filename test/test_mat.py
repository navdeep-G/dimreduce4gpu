import mat
import numpy as np
from collections import namedtuple
from ksvd import ApproximateKSVD


def approx_equal(a, b, t):
    if b - a < abs(b * t):
        return True
    else:
        return False


Case = namedtuple('Case', ['n', 'm', 'dict_size', 'target_sparsity', 'iterations', 'sparse_updater', 'dict_updater'])

cases = []
cases.append(Case(10, 5, 6, 3, 20, 'mp', 'gd'))
cases.append(Case(1000, 30, 40, 12, 20, 'mp', 'gd'))
cases.append(Case(500, 300, 400, 120, 20, 'mp', 'gd'))

dict_updaters = [('gd', 0.5), ('aksvd', 0.2), ('agd', 0.2)]
sparse_updaters = [('mp', 0.5), ('omp2', 0.2), ('omp5', 0.2), ('omp10', 0.2)]


def check_sparsity(S, target_sparsity, t):
    sparsity = np.count_nonzero(S) / S.shape[1]
    assert (approx_equal(target_sparsity, sparsity, t))


# Check for any zero length columns in D
def check_D(D):
    for column in D.T:
        assert (np.linalg.norm(column) > 0)


# Evaluate if the mat reconstruction error is within tolerance t of ksvd or better
def evaluate_case(c, t):
    print(c)
    X = mat.generate_X(c.m, c.n, c.dict_size, c.target_sparsity, 13)
    ksvd_result = KSVD(X.transpose(), c.dict_size, c.target_sparsity, c.iterations, print_interval=20,
                       enable_printing=False, enable_threading=True, D_init='random')
    rmse1 = mat.rmse(X, ksvd_result[0].transpose(), ksvd_result[1].transpose())
    mat_result = mat.sparse_code(X, c.dict_size, c.target_sparsity, c.iterations, sparse_updater=c.sparse_updater,
                                 dict_updater=c.dict_updater)
    print(mat_result[2])
    rmse2 = mat.rmse(X, mat_result[0], mat_result[1])
    print('ksvd rmse:' + str(rmse1))
    print('mat rmse:' + str(rmse2))
    assert (approx_equal(rmse1, rmse2, t))

    check_sparsity(mat_result[1], c.target_sparsity, 0.1)
    check_D(mat_result[0])


def test_sparse_updaters():
    for c in cases:
        for u in sparse_updaters:
            c = c._replace(sparse_updater=u[0])
            evaluate_case(c, u[1])


def test_dict_updaters():
    for c in cases:
        for u in dict_updaters:
            c = c._replace(dict_updater=u[0])
            evaluate_case(c, u[1])
