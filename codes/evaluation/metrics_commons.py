import cupy as cp


def sort_topk_matrix_row_by_another_matrix(return_array, sorting_array, topk):
    """
    Sorts a matrix row-wise based on other matrix values

    Args:
        return_array (cupy.core.core.ndarray): A 2D matrix with values to be returned
        sorting_array (cupy.core.core.ndarray): A 2D matrix with the values to sort (row-wise)

    Returns:
        A 2D matrix with the same dimension of the input matrices, with the rows of "return_array"
        sorted row-wise by "sorting_array"
    """

    if return_array.shape != sorting_array.shape:
        raise ValueError("Dimensions of the arrays be exactly the same!")

    if len(return_array.shape) != 2:
        raise ValueError("The matrices should have 2 dimensions.")

    rows_indexes = cp.arange(sorting_array.shape[0]).reshape(-1, 1)

    n_cols = return_array.shape[1]
    top_k_descending = max(n_cols - topk, 0)
    # Partitioning into two vectors at the k-th position (values of the left partition will be lower than the value at k-th
    # and values on the right partition will be equal or greather the the k-th value)
    # P.s. Sounds like cuPy actually implements partition as a full sort, but is very efficient either in that case, as described in this issue: https://github.com/cupy/cupy/issues/478
    col_indexes_by_row_partitioned_topk = cp.argpartition(
        sorting_array, kth=top_k_descending, axis=1
    )
    sorting_values_by_row_partitioned_topk = sorting_array[
        rows_indexes, col_indexes_by_row_partitioned_topk
    ]
    values_by_row_partitioned_topk = return_array[
        rows_indexes, col_indexes_by_row_partitioned_topk
    ]

    # Resorting only the top-k items (because partition does not ensure that the left partition is sorted)
    col_indexes_by_row_sorted_topk = top_k_descending + cp.argsort(
        sorting_values_by_row_partitioned_topk[:, top_k_descending:], axis=1
    )
    values_by_row_sorted_topk = cp.hstack(
        [
            values_by_row_partitioned_topk[:, :top_k_descending],
            values_by_row_partitioned_topk[rows_indexes, col_indexes_by_row_sorted_topk],
        ]
    )

    # Reversing the matrix so that highest values come in the top-k position
    values_by_row_sorted_topk = cp.flip(values_by_row_sorted_topk, axis=1)

    return values_by_row_sorted_topk


def is_in_2d_rowwise(matrix, in_matrix):
    """
    Returns a binary matrix with the same dimension of the values matrix, 
    with "ones" only in positions where the value is in the corresponding "in_values" vector (row-wise)

    Args:
        matrix (cupy.core.core.ndarray): A 2D matrix
        in_matrix (cupy.core.core.ndarray): A 2D matrix with the values to look for (row-wise)

    Returns:
        A binary 2D matrix with the same dimension than the "values" matrix, with "ones" 
        only for values which are in the corresponding "in_values" vector (row-wise)
    """
    if not (len(matrix.shape) == len(in_matrix.shape) == 2):
        raise Exception("Both matrices should have exactly two dimensions (rank 2)")

    if not (matrix.shape[0] == in_matrix.shape[0]):
        raise Exception("Both matrices should have the same number of rows (axis=0)")

    matrix_expanded = cp.expand_dims(matrix, axis=2)
    in_matrix_expanded = cp.expand_dims(in_matrix, axis=1)
    result = (matrix_expanded == in_matrix_expanded).astype(cp.int8).sum(axis=2)
    return result
