from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import cython, random
import numpy as numpy
#cimport numpy as numpy
from cython.parallel import parallel, prange
from libc.stdlib cimport rand, srand
from libcpp.set cimport set as c_set

@cython.boundscheck(False)
def sample_warp(train_dict, Py_ssize_t n_items, Py_ssize_t warp_count, Py_ssize_t batch_size):
    cdef Py_ssize_t index, i, j , row, u, pointer_to_triplet, j_neg, start_row_in_triplet, offset, j_pos, pointer_to_pair

    # create pairs that contain all the positive (user,item) pairs
    pos_sample_count = sum([len(items) for items in train_dict.values()])
    n_users = numpy.max(train_dict.keys())+1
    # a <n_pos_sample, 2> matrix
    cdef long [:, :] pairs_view = numpy.zeros((pos_sample_count, 2), dtype="int64")
    # an array of set
    cdef c_set[long]** pos_items = <c_set[long]**> PyMem_Malloc(n_users * sizeof(void*));
    index = 0
    for u, items in train_dict.items():
        pos_items[u] = new c_set[long]()
        for item in items:
            pos_items[u].insert(item)
            pairs_view[index, 0] = u
            pairs_view[index, 1] = item
            index += 1
    orders = numpy.arange(len(pairs_view))
    # shuffle the orders array
    numpy.random.seed()
    numpy.random.shuffle(orders)
    cdef long[:] orders_view = orders
    # reseed c random generator
    srand(numpy.random.randint(1000000))
    # the sampled triplets we will return at the end
    cdef long [:, :] sampled_triplets_view = numpy.zeros((batch_size * warp_count, 3), dtype="int64")

    index = 0
    try:
        while True:
            # when we do not have enough remaining pairs to sample, re-shuffle the pairs again and reset index,
            if index + batch_size > pairs_view.shape[0]:
                numpy.random.seed()
                numpy.random.shuffle(orders)
                orders_view = orders
                index = 0
            with nogil:
                for offset in range(batch_size):
                    row = index + offset
                    pointer_to_pair = orders_view[row]
                    u = pairs_view[pointer_to_pair, 0]
                    j_pos = pairs_view[pointer_to_pair, 1]
                    start_row_in_triplet = offset * warp_count
                    for pointer_to_triplet in range(start_row_in_triplet, start_row_in_triplet + warp_count):
                        j_neg = rand() % n_items
                        while pos_items[u].count(j_neg) :
                            j_neg = rand() % n_items
                        sampled_triplets_view[pointer_to_triplet,0] = u
                        sampled_triplets_view[pointer_to_triplet,1] = j_pos
                        sampled_triplets_view[pointer_to_triplet,2] = j_neg
            index += batch_size
            yield numpy.asarray(sampled_triplets_view)
    finally:
        for u, items in train_dict.items():
            PyMem_Free(pos_items[u])
        PyMem_Free(pos_items)