import numpy
from scipy.sparse import dok_matrix


class WarpSampler(object):
    """
    A generator that generate tuples: user-positive-item pairs, negative-items

    of shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """

    def __init__(self, user_item_matrix, batch_size=10000, n_negative=10):
        self.user_item_matrix = dok_matrix(user_item_matrix)
        self.user_item_pairs = numpy.asarray(self.user_item_matrix.nonzero()).T
        self.batch_size = batch_size
        self.n_negative = n_negative

    @property
    def sample(self):
        while True:
            numpy.random.shuffle(self.user_item_pairs)
            for i in range(int(len(self.user_item_pairs) / self.batch_size)):
                user_positive_items_pairs = self.user_item_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]
                # Sample random items as negative samples. Note that, in this way, we may potentially sample a few items
                # that are actually in the positive item set for the user. However, the chance is lower given a large
                # enough item get enough.

                negative_samples = numpy.random.randint(
                    0,
                    self.user_item_matrix.shape[1],
                    size=(self.batch_size, self.n_negative))

                yield user_positive_items_pairs, negative_samples

    def next_batch(self):
        return self.sample.__next__()