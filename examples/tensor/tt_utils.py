import numpy as np
import torch
import warnings
from typing import Tuple, List, Iterable, Union

def validate_tt_rank(
    tensor_shape: Tuple[int],
    rank: Union[int, Tuple[int], List[int], float] = "same",
    constant_rank: bool = False,
    rounding: str = "round",
    allow_overparametrization: bool = False,
    per_decomp_rank_ratio_limit: float = 1.0,
    strict_rank_ratio_limit: bool = False
):
    """Returns the rank of a TT Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, tuple, list, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int or tuple or list, just returns rank
    constant_rank : bool, default is False
        * if True, the *same* rank will be chosen for each modes
        * if False (default), the rank of each mode will be proportional to the corresponding tensor_shape

        *used only if rank == 'same' or 0 < rank <= 1*

    rounding = {'round', 'floor', 'ceil'}

    allow_overparametrization : bool, default is True
        if False, the rank must be realizable through iterative application of SVD
        (used in tensorly.decomposition.tensor_train)

    Returns
    -------
    rank : int tuple
        rank of the decomposition
    """
    if rounding == "ceil":
        rounding_fun = np.ceil
    elif rounding == "floor":
        rounding_fun = np.floor
    elif rounding == "round":
        rounding_fun = np.round
    else:
        raise ValueError(f"Rounding should be round, floor or ceil, but got {rounding}")

    if rank == "same":
        rank = float(1)

    if isinstance(rank, float) and constant_rank:
        # Choose the *same* rank for each mode
        n_param_tensor = np.prod(tensor_shape) * rank
        order = len(tensor_shape)

        if order == 2:
            rank = (1, n_param_tensor / (tensor_shape[0] + tensor_shape[1]), 1)
            warnings.warn(
                f"Determining the tt-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor."
            )

        # R_k I_k R_{k+1} = R^2 I_k
        a = np.sum(tensor_shape[1:-1])

        # Border rank of 1, R_0 = R_N = 1
        # First and last factor of size I_0 R and I_N R
        b = np.sum(tensor_shape[0] + tensor_shape[-1])

        # We want the number of params of decomp (=sum of params of factors)
        # To be equal to c = \prod_k I_k
        c = -n_param_tensor
        delta = np.sqrt(b**2 - 4 * a * c)

        # We get the non-negative solution
        solution = int(rounding_fun((-b + delta) / (2 * a)))
        rank = rank = (1,) + (solution,) * (order - 1) + (1,)

    elif isinstance(rank, float):
        # Choose a rank proportional to the size of each mode
        # The method is similar to the above one for constant_rank == True
        order = len(tensor_shape)
        avg_dim = [
            (tensor_shape[i] + tensor_shape[i + 1]) / 2 for i in range(order - 1)
        ]
        if len(avg_dim) > 1:
            a = sum(
                avg_dim[i - 1] * tensor_shape[i] * avg_dim[i]
                for i in range(1, order - 1)
            )
        else:
            warnings.warn(
                f"Determining the tt-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor."
            )
            a = avg_dim[0] ** 2 * tensor_shape[0]
        b = tensor_shape[0] * avg_dim[0] + tensor_shape[-1] * avg_dim[-1]
        c = -np.prod(tensor_shape) * rank
        delta = np.sqrt(b**2 - 4 * a * c)

        # We get the non-negative solution
        fraction_param = (-b + delta) / (2 * a)
        rank = tuple([max(int(rounding_fun(d * fraction_param)), 1) for d in avg_dim])
        rank = (1,) + rank + (1,)

    else:
        # Check user input for potential errors
        n_dim = len(tensor_shape)
        if isinstance(rank, int):
            rank = [1] + [rank] * (n_dim - 1) + [1]
        elif n_dim + 1 != len(rank):
            message = f"Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {len(rank)} while tl.ndim(tensor) + 1  = {n_dim+1}"
            raise (ValueError(message))

        # Initialization
        if rank[0] != 1:
            message = "Provided rank[0] == {} but boundary conditions dictate rank[0] == rank[-1] == 1.".format(
                rank[0]
            )
            raise ValueError(message)
        if rank[-1] != 1:
            message = "Provided rank[-1] == {} but boundary conditions dictate rank[0] == rank[-1] == 1.".format(
                rank[-1]
            )
            raise ValueError(message)

    if allow_overparametrization:
        return list(rank)
    else:
        validated_rank = [1]
        for i, s in enumerate(tensor_shape[:-1], 1):
            n_row = int(validated_rank[i-1] * s)
            n_column = np.prod(tensor_shape[(i) :])  # n_column of unfolding
            if strict_rank_ratio_limit:
                validated_rank_i = min(int(min(n_row, n_column)*per_decomp_rank_ratio_limit), rank[i])
            else:
                validated_rank_i = min(n_row, n_column, rank[i])
                if not rank[i] <= validated_rank_i:
                    if rank[i] <= min(n_row, n_column):
                        validated_rank_i = rank[i]
                    else:
                        validated_rank_i = int(validated_rank_i * per_decomp_rank_ratio_limit)
            validated_rank.append(validated_rank_i)
        validated_rank.append(1)

        return validated_rank