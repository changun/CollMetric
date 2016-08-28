from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport rand, RAND_MAX, srand
import cython
from libcpp.set cimport set as c_set
from libcpp.vector cimport vector
import numpy as np
import time

@cython.boundscheck(False)
def sample(dict train_dict, dict exclude_dict, set exclude_items_set, int n_items, int per_user_sample):
    cdef int[2]** dat

    cdef int n_users = len(train_dict.keys())
    cdef int pos
    cdef int i
    cdef Py_ssize_t j, base, neg
    cdef c_set[int]* excludes
    cdef index = 0

    srand(np.random.randint(1000000))

    cdef c_set[int]** exclude_items = <c_set[int]**> PyMem_Malloc(n_users * sizeof(void*));
    cdef vector[int] n_pos_for_user = [-1] * n_users
    triplets = np.zeros(((n_users * per_user_sample), 4), dtype="int64")
    dat = <int[2]**> PyMem_Malloc(n_users * sizeof(int[2]*))
    cdef int sample_id = 0
    cdef vector[int] unique_j
    unique_i = np.asarray(list(train_dict.keys()))
    for i, (user, items) in enumerate(train_dict.items()):
        n_pos_for_user[i] = len(items)
        dat[i] = <int[2]*> PyMem_Malloc(len(items) * sizeof(int[2]))
        for j, item in enumerate(items):
            dat[i][j][0] = item
            dat[i][j][1] = sample_id
            sample_id += 1
        for j in xrange(per_user_sample):
            triplets[i*per_user_sample + j][0] = user
        exclude_items[i] = new c_set[int]()
        for item in items:
            exclude_items[i].insert(item)
        if exclude_dict is not None and user in exclude_dict:
            for item in exclude_dict[user]:
                exclude_items[i].insert(item)
    cdef j_in_iteration = [-1] * n_items
    try:
        iteration = 0
        while True:

            for i in xrange(n_users):
                base = i * per_user_sample
                for j in range(base, base + per_user_sample):
                    sample = dat[i][rand() % n_pos_for_user[i]]
                    pos = sample[0]
                    triplets[j][1] = pos
                    triplets[j][3] = sample[1]
                    neg = rand() % n_items
                    while exclude_items[i].count(neg) == 1 or neg in exclude_items_set:
                        neg = rand() % n_items
                    triplets[j][2] = neg
                    if j_in_iteration[pos] != iteration:
                        j_in_iteration[pos]=iteration
                        unique_j.push_back(pos)
                    if j_in_iteration[neg] != iteration:
                        j_in_iteration[neg]=iteration
                        unique_j.push_back(neg)

            yield triplets
            unique_j.clear()
            iteration += 1
    finally:
        for i in range(n_users):
            PyMem_Free(exclude_items[i])
            PyMem_Free(dat[i])
        PyMem_Free(exclude_items)
        PyMem_Free(dat)




