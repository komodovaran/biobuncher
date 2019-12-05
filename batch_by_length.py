import numpy as np
import sklearn.utils

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def batch_by_length(X, indices = None, max_batch_size = 32):
    """
    Batches a list of variable-length samples into equal-sized tensors
    to speed up training.
    Indices maintain the same order of unravelled batches
    """
    if indices is None:
        indices = np.arange(0, len(X), 1)

    lengths = [len(xi) for xi in X]
    length_brackets = np.unique(lengths)

    # initialize empty batches for each length
    length_batches = [[] for _ in range(len(length_brackets))]
    idx_batches = [[] for _ in range(len(length_brackets))]
    if not len(length_batches) == len(length_brackets):
        raise ValueError

    # Go through each sample and find out where it belongs
    for i in range(len(X)):
        xi = X[i]
        idx = indices[i]

        # Find out which length bracket it belongs to
        (belongs_to,) = np.where(len(xi) == length_brackets)
        belongs_to = belongs_to[0]

        # Place sample there
        length_batches[belongs_to].append(xi)
        idx_batches[belongs_to].append(idx)

    # Break into smaller chunks so that a batch is at most max_batch_size
    dataset = []
    index = []
    for j in range(len(length_batches)):
        sub_batch = list(chunks(length_batches[j], max_batch_size))
        sub_idx = list(chunks(idx_batches[j], max_batch_size))

        for k in range(len(sub_batch)):
            dataset.append(sub_batch[k])
            index.append(sub_idx[k])

    # Now transform each batch to a tensor
    dataset = [np.array(batch) for batch in dataset]
    index_set = [np.array(index_batch) for index_batch in index]
    return dataset, index_set

if __name__ == "__main__":
    X = []
    for _ in range(10):
        xi = np.zeros((np.random.randint(10, 30), 2))
        X.append(xi)
    indices = np.arange(0, len(X), 1)

    max_size = 128
    dataset, indexset = batch_by_length(X, max_batch_size = max_size, indices = indices)
    print("batch 3, sample 0 has length: ", len(dataset[3][0]))
    print("batch 3, ID 0 is, ", indexset[3][0])
    print("original data length retrieved from ID ", len(X[indexset[3][0]]))
    quit()

    for b in dataset:
        if len(b) == 1:
            print(type(b))
            print(b.shape)