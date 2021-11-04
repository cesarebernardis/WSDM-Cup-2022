import numpy as np
import scipy.sparse as sps


def invert_dictionary(id_to_index):

    index_to_id = {}

    for id in id_to_index.keys():
        index = id_to_index[id]
        index_to_id[index] = id

    return index_to_id


def estimate_sparse_size(num_rows, topK):
    """
    :param num_rows: rows or colum of square matrix
    :param topK: number of elements for each row
    :return: size in Byte
    """

    num_cells = num_rows*topK
    sparse_size = 4*num_cells*2 + 8*num_cells

    return sparse_size


def seconds_to_biggest_unit(time_in_seconds, data_array=None):

    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value/conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            if data_array is not None:
                data_array /= conversion_factor[unit_index][1]

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True

    if data_array is not None:
        return new_time_value, new_time_unit, data_array

    else:
        return new_time_value, new_time_unit


def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """


    if sps.issparse(X):
        if format == 'csc' and not isinstance(X, sps.csc_matrix):
            return X.tocsc().astype(dtype)
        elif format == 'csr' and not isinstance(X, sps.csr_matrix):
            return X.tocsr().astype(dtype)
        elif format == 'coo' and not isinstance(X, sps.coo_matrix):
            return X.tocoo().astype(dtype)
        elif format == 'dok' and not isinstance(X, sps.dok_matrix):
            return X.todok().astype(dtype)
        elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
            return X.tobsr().astype(dtype)
        elif format == 'dia' and not isinstance(X, sps.dia_matrix):
            return X.todia().astype(dtype)
        elif format == 'lil' and not isinstance(X, sps.lil_matrix):
            return X.tolil().astype(dtype)
        return X.astype(dtype)
    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        X.sort_indices()
        return check_matrix(X, format=format, dtype=dtype)


def reshapeSparse(sparseMatrix, newShape):

    if sparseMatrix.shape[0] > newShape[0] or sparseMatrix.shape[1] > newShape[1]:
        ValueError("New shape cannot be smaller than SparseMatrix. SparseMatrix shape is: {}, newShape is {}".format(
            sparseMatrix.shape, newShape))


    sparseMatrix = sparseMatrix.tocoo()
    newMatrix = sps.csr_matrix((sparseMatrix.data, (sparseMatrix.row, sparseMatrix.col)), shape=newShape)

    return newMatrix


# Matrices can get very big, use as few space as possible on disk

def smaller_uint(array):
    if array.max() > 4294967295:
        return np.uint64
    elif array.max() > 65535:
        return np.uint32
    elif array.max() > 255:
        return np.uint16
    return np.uint8


def smaller_int(array):
    if array.max() > 2147483647 or array.min() < -2147483648:
        return np.int64
    elif array.max() > 32767 or array.min() < -32768:
        return np.int32
    elif array.max() > 127 or array.min() < -128:
        return np.int16
    return np.int8


def smaller_datatype(array):
    if issubclass(array.dtype.type, np.floating):
        return np.float32
    elif array.min() < 0:
        return smaller_int(array)
    return smaller_uint(array)


def save_compressed_csr_matrix(matrix, filename, truncated=True):
    print("Saving {}".format(filename))
    kwargs = {
        "shape": np.array([matrix.shape[0], matrix.shape[1]], dtype=np.int32),
        "indptr": matrix.indptr.astype(smaller_uint(matrix.indptr)),
        "indices": matrix.indices.astype(smaller_uint(matrix.indices))
    }
    if not truncated:
        kwargs["data"] = matrix.data.astype(smaller_datatype(matrix.data))
    np.savez_compressed(filename, **kwargs)



def load_compressed_csr_matrix(filename, truncated=True):
    print("Loading {}".format(filename))
    arrays = np.load(filename)
    indices = arrays['indices']
    indptr = arrays['indptr']
    if truncated:
        data = np.ones(indices.size).astype(np.int8)
    else:
        data = arrays['data']
    return sps.csr_matrix(
        (data, indices, indptr),
        shape=(arrays['shape'][0], arrays['shape'][1])
    ).astype(data.dtype)

