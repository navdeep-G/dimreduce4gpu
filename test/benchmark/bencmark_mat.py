import mat, time
from collections import namedtuple
from ksvd import KSVD
import matplotlib.pyplot as plt
import numpy as np
import platform

Parameters = namedtuple('Case', ['n', 'm', 'dict_size', 'target_sparsity', 'iterations', 'dict_updater'])

sparse_updaters = ['mp', 'omp2', 'omp5', 'omp10']
#sparse_updaters = ['mp', 'omp2','omp5']


def pareto():
    #p = Parameters(1000, 200, 300, 50, 10, 'agd')
    p = Parameters(2000, 200, 300, 50, 10, 'gd')
    #p = Parameters(1000, 80, 100, 20, 10, 'agd')
    labels = sparse_updaters[:]
    labels.append('ksvd')

    iter_times = []
    iter_accuracy = []
    iterations = 20
    for iter in range(0, iterations):
        times = []
        accuracy = []
        X = mat.generate_X(p.m, p.n, p.dict_size, p.target_sparsity, iter)
        # Warmup
        mat.sparse_code(X, p.dict_size, p.target_sparsity, p.iterations, metric='pyksvd',
                        dict_updater=p.dict_updater)
        for updater in sparse_updaters:
            tmp = time.time()
            result = mat.sparse_code(X, p.dict_size, p.target_sparsity, p.iterations, metric='pyksvd',
                                     dict_updater=p.dict_updater, sparse_updater=updater)
            times.append(time.time() - tmp)
            accuracy.append(mat.rmse(X, result[0], result[1]))

        tmp = time.time()
        ksvd_result = KSVD(X.transpose(), p.dict_size, p.target_sparsity, p.iterations, print_interval=5, enable_printing=True,
             enable_threading=True, D_init='random')
        times.append(time.time() - tmp)
        accuracy.append(mat.rmse(X,ksvd_result[0].transpose(), ksvd_result[1].transpose()))
        iter_times.append(times)
        iter_accuracy.append(accuracy)

    #Average iterations
    times = [float(sum(col)) / len(col) for col in zip(*iter_times)]
    accuracy = [float(sum(col)) / len(col) for col in zip(*iter_accuracy)]

    #Plot
    plt.gca().set_position((.1, .3, .8, .6))
    plt.scatter(times, accuracy)
    plt.xlabel('time(s)')
    plt.ylabel('rmse')
    plt.title('Accuracy vs. Execution Time for Sparse Updaters')
    for label, x, y in zip(labels, times, accuracy):
        plt.annotate(label, xy=(x,y))
    plt.figtext(.02,.02, str(p))
    plt.show()

pareto()



