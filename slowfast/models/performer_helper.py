#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from einops import rearrange
import math
import numpy as np
import torch

BIG_CONSTANT = 1e8


def create_projection_matrix(
    m, d, seed=0, scaling=0, struct_mode=False, device='cuda', dtype=torch.float32
):
    r"""Constructs the matrix of random projections.
    Constructs a matrix of random orthogonal projections. Each projection vector
    has direction chosen uniformly at random and either deterministic length
    \sqrt{d} or length taken from the \chi(d) distribution (in the latter case
    marginal distributions of the projections are d-dimensional Gaussian vectors
    with associated identity covariance matrix).
    Args:
        m: number of random projections.
        d: dimensionality of each random projection.
        seed: random seed used to construct projections.
        scaling: 1 if all the random projections need to be renormalized to have
        length \sqrt{d}, 0 if the lengths of random projections should follow
        \chi(d) distribution.
        struct_mode: if True then products of Givens rotations will be used to
        construct random orthogonal matrix. This bypasses Gram-Schmidt
        orthogonalization.
    Returns:
        The matrix of random projections of the shape [m, d].
    """
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        if struct_mode:
            q = create_products_of_givens_rotations(d, seed)
        else:
            torch.manual_seed(current_seed)
            unstructured_block = torch.randn((d, d), device = device)
            q, _ = torch.qr(unstructured_block.cpu(), some=True)
            q = q.to(device=device)
            q = q.t()
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        if struct_mode:
            q = create_products_of_givens_rotations(d, seed)
        else:
            torch.manual_seed(seed)
            unstructured_block = torch.randn((d, d), device = device)
            q, _ = torch.qr(unstructured_block.cpu(), some=True)
            q = q.to(device=device)
            q = q.t()
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)
    current_seed += 1

    if scaling == 0:
        torch.manual_seed(current_seed)
        multiplier = torch.norm(torch.randn((m, d), device = device), p='fro', dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(float(d)) * torch.ones((m)).to(device=device)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
    """Computes random features for the softmax kernel using FAVOR+ mechanism.
    Computes random features for the softmax kernel using FAVOR+ mechanism from
    https://arxiv.org/pdf/2009.14794.pdf.
    Args:
        data: input data tensor of the shape [B, L, H, D], where: B - batch
        dimension, L - attention dimensions, H - heads, D - features.
        is_query: indicates whether input data is a query oor key tensor.
        projection_matrix: random Gaussian matrix of shape [M, D], where M stands
        for the number of random features and each D x D sub-block has pairwise
        orthogonal rows.
        numerical_stabilizer: small positive constant for numerical stability.
    Returns:
        Corresponding kernel feature map.
    """
    data_normalizer = (data.shape[-1] ** -0.25)
    data = data_normalizer * data
    ratio = (projection_matrix.shape[0] ** -0.5)

    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
    diag_data = data ** 2
    diag_data = torch.sum(
        diag_data, dim=data.ndim - 1)
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(data.ndim - 1)
    last_dims_t = (len(data_dash.shape) - 1,)
    attention_dims_t = (len(data_dash.shape) - 3,)
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - data_dash.amax(
                last_dims_t, True)) + numerical_stabilizer)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - data_dash.amax(
                last_dims_t + attention_dims_t, True)) + numerical_stabilizer)

    # return data_dash.type_as(data)
    return data_dash


def create_products_of_givens_rotations(dim, seed):
    r"""Constructs a 2D-tensor which is a product of Givens random rotations.
    Constructs a 2D-tensor of the form G_1 * ... * G_k, where G_i is a Givens
    random rotation. The resulting tensor mimics a matrix taken uniformly at
    random form the orthogonal group.
    Args:
        dim: number of rows/columns of the resulting 2D-tensor.
        seed: random seed.
    Returns:
        The product of Givens random rotations.
    """
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.sin(
            random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(
            random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return q.detach().type(torch.float32)


def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
    """Computes features for the ReLU-kernel.
    Computes random features for the ReLU kernel from
    https://arxiv.org/pdf/2009.14794.pdf.
    Args:
        data: input data tensor of the shape [B, L, H, D], where: B - batch
        dimension, L - attention dimensions, H - heads, D - features.
        is_query: indicates whether input data is a query oor key tensor.
        projection_matrix: random Gaussian matrix of shape [M, D], where M stands
        for the number of random features and each D x D sub-block has pairwise
        orthogonal rows.
        numerical_stabilizer: small positive constant for numerical stability.
    Returns:
        Corresponding kernel feature map.
    """
    del is_query
    if projection_matrix is None:
        return torch.nn.functional.relu(data) + numerical_stabilizer
    else:
        ratio = 1.0 / math.sqrt(torch.tensor(projection_matrix.shape[0]).float())
        data_dash = ratio * torch.einsum("blhd,md->blhm", data, projection_matrix)
        return torch.nn.functional.relu(data_dash) + numerical_stabilizer


def noncausal_numerator(qs, ks, vs):
    """Computes not-normalized FAVOR noncausal attention AV.
    Args:
        qs: query_prime tensor of the shape [L,B,H,M].
        ks: key_prime tensor of the shape [L,B,H,M].
        vs: value tensor of the shape [L,B,H,D].
    Returns:
        Not-normalized FAVOR noncausal attention AV.
    """
    kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
    return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
    """Computes FAVOR normalizer in noncausal attention.
    Args:
        qs: query_prime tensor of the shape [L,B,H,M].
        ks: key_prime tensor of the shape [L,B,H,M].
    Returns:
        FAVOR normalizer in noncausal attention.
    """
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
    return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)


def favor_attention(query,
                    key,
                    value,
                    projection_matrix=None):
    """Computes FAVOR normalized attention.
    Args:
        query: query tensor.
        key: key tensor.
        value: value tensor.
        kernel_transformation: transformation used to get finite kernel features.
        projection_matrix: projection matrix to be used.
    Returns:
        FAVOR normalized attention.
    """
    query_prime = softmax_kernel_transformation(query, True,
                                        projection_matrix)  # [B,L,H,M]
    key_prime = softmax_kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
    query_prime = query_prime.permute(1, 0, 2, 3)  # [L,B,H,M]
    key_prime = key_prime.permute(1, 0, 2, 3)  # [L,B,H,M]
    value = value.permute(1, 0, 2, 3)  # [L,B,H,D]
    av_attention = noncausal_numerator(query_prime, key_prime, value)
    attention_normalizer = noncausal_denominator(query_prime, key_prime)
    av_attention = av_attention.permute(1, 0, 2, 3)
    attention_normalizer = attention_normalizer.permute(1, 0, 2)
    attention_normalizer = attention_normalizer.unsqueeze(len(attention_normalizer.shape))
    return av_attention / attention_normalizer


def get_attention(
    query, key, value, numerical_stabilizer=0.001, 
    projection_matrix_type=None, nb_random_features=12, cache=None
):
    if projection_matrix_type is None:
        projection_matrix = None
    else:
        dim = query.shape[-1]
        seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT))
        seed = torch.tensor(seed)
        projection_matrix = create_projection_matrix(
            nb_random_features, dim, seed=seed)
    

    if cache is not None:
        # Combine cached keys and values with new keys and values.
        if decode_loop_step is not None:
            cache_k_shape = cache["k"].shape.as_list()
            indices = tf.reshape(
                tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
                [1, cache_k_shape[1], 1, 1])
            key = cache["k"] + key * indices
            cache_v_shape = cache["v"].shape.as_list()
            indices = tf.reshape(
                tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
                [1, cache_v_shape[1], 1, 1])
            value = cache["v"] + value * indices
        else:
            key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
            value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

        # Update cache
        cache["k"] = key
        cache["v"] = value

    attention_output = favor_attention(query, key, value, projection_matrix)
    return attention_output


def test_softmax_noncausal_attention_block_output():
    batch_size = 1
    # length = 10000
    length = 1000
    num_heads = 1
    dim = 8
    num_random_features = 100
    query = torch.randn(batch_size, length, num_heads, dim).cuda()
    key = torch.randn(batch_size, length, num_heads, dim).cuda()
    value = torch.randn(batch_size, length, num_heads, dim).cuda()
    projection_matrix = create_projection_matrix(num_random_features, dim)
    attention_block_output = favor_attention(query, key, value, projection_matrix)

    attention_scores = torch.einsum(
        "bxhd,byhd->bxyh", torch.multiply(query, 1.0 / math.sqrt(float(dim))), key)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=2)
    exact_attention_block_output = torch.einsum("bxyh,byhd->bxyh", attention_scores, value)
    max_error = 0.5
    error = torch.abs(exact_attention_block_output - attention_block_output)
    print(torch.max(error), max_error)

    
    q_prime = softmax_kernel_transformation(
        query, is_query=True, projection_matrix=projection_matrix)
    k_prime = softmax_kernel_transformation(
        key, is_query=False, projection_matrix=projection_matrix)
    
    scale = 1.0 / float(math.sqrt(dim))
    qpkp = torch.einsum('b h p r, b h q r -> b h p q', q_prime, k_prime)
    q_, k_ = map(lambda t: rearrange(t, f'b p h d -> b h p d'), (query, key))
    qkexp = torch.einsum('b h p d, b h q d -> b h p q', q_ * scale, k_).exp()
    print('max absolute error A: ', (qkexp - qpkp).abs().max())
    print('mean absolute error A: ', (qkexp - qpkp).abs().mean())
    print((qkexp.isclose(qpkp).sum() / qpkp.numel()) * 100.0)
    assert (qpkp <= 0.0).sum() == 0.0


if __name__ == "__main__":
    test_softmax_noncausal_attention_block_output()