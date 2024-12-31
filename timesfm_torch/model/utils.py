import torch

def mean_std(inputs):
    """
    inputs: a tensor of shape (bs, patch_len, patch_size)
    """
    assert len(inputs.shape) == 3
    inputs_mean = inputs.mean(dim=(1, 2))
    inputs_std = inputs.std(dim=(1, 2))
    return inputs_mean, inputs_std

def forward_transform(inputs, patched_pads, tolerance, pad_val):
    """Input is of shape [B, N, P]."""
    mu, sigma = _masked_mean_std(inputs, patched_pads)
    sigma = torch.where(
        sigma < tolerance,
        torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
        sigma,
    )

    # Normalize each patch
    outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
    outputs = torch.where(
        torch.abs(inputs - pad_val) < tolerance,
        torch.tensor(pad_val,
                    dtype=outputs.dtype,
                    device=outputs.device),
        outputs,
    )
    return outputs, (mu, sigma)

def reverse_transform(outputs, stats):
    """Output is of shape [B, N, P, Q]."""
    mu, sigma = stats
    return outputs * sigma[:, None, None, None] + mu[:, None, None, None]


def _masked_mean_std(
    inputs: torch.Tensor,
    padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean and standard deviation of `inputs` across axis 1.

    It excludes values where `padding` is 1.

    Args:
        inputs: A PyTorch tensor of shape [b, n, p].
        padding: A PyTorch tensor of shape [b, n, p] with values 0 or 1.

    Returns:
        A tuple containing the mean and standard deviation.
        We return the statistics of the first patch with more than three non-padded
        values.
    """
    # Selecting the first patch with more than 3 unpadded values.
    pad_sum = torch.sum(1 - padding, dim=2)

    def _get_patch_index(arr: torch.Tensor):
        indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
        row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
        return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

    patch_indices = _get_patch_index(pad_sum)
    bidxs = torch.arange(inputs.shape[0])

    arr = inputs[bidxs, patch_indices, :]
    pad = padding[bidxs, patch_indices, :]

    # Create a mask where padding is 0
    mask = 1 - pad

    # Calculate the number of valid elements
    num_valid_elements = torch.sum(mask, dim=1)
    num_valid_elements = torch.where(
        num_valid_elements == 0,
        torch.tensor(1,
                    dtype=num_valid_elements.dtype,
                    device=num_valid_elements.device),
        num_valid_elements,
    )

    # Calculate the masked sum and squared sum
    masked_sum = torch.sum(arr * mask, dim=1)
    masked_squared_sum = torch.sum((arr * mask)**2, dim=1)

    # Calculate the masked mean and standard deviation
    masked_mean = masked_sum / num_valid_elements
    masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
    masked_var = torch.where(
        masked_var < 0.0,
        torch.tensor(0.0, dtype=masked_var.dtype, device=masked_var.device),
        masked_var,
    )
    masked_std = torch.sqrt(masked_var)

    return masked_mean, masked_std


def shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """Shifts rows of seq based on the first 0 in each row of the mask.

    Args:
        mask: mask tensor of shape [B, N]
        seq: seq tensor of shape [B, N, P]

    Returns:
        Returns the shifted sequence.
    """
    batch_size, num_seq, feature_dim = seq.shape

    new_mask: torch.BoolTensor = mask == 0

    # Use argmax to find the first True value in each row
    indices = new_mask.to(torch.int32).argmax(dim=1)

    # Handle rows with all zeros
    indices[~new_mask.any(dim=1)] = -1

    # Create index ranges for each sequence in the batch
    idx_range = (torch.arange(num_seq).to(
        seq.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1,
                                                        feature_dim))

    # Calculate shifted indices for each element in each sequence
    shifted_idx = (idx_range - indices[:, None, None]) % num_seq

    # Gather values from seq using shifted indices
    shifted_seq = seq.gather(1, shifted_idx)

    return shifted_seq

def get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
    """Returns a large negative value for the given dtype."""
    if dtype.is_floating_point:
        dtype_max = torch.finfo(dtype).max
    else:
        dtype_max = torch.iinfo(dtype).max
    return torch.tensor(-0.7 * dtype_max, dtype=dtype)

def convert_paddings_to_mask(
    paddings: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Converts binary paddings to a logit mask ready to add to attention matrix.

    Args:
        paddings: binary torch.Tensor of shape [B, T], with 1 denoting padding
            token.
        dtype: data type of the input.

    Returns:
        A torch.Tensor of shape [B, 1, 1, T] ready to add to attention logits.
    """
    attention_mask = paddings.detach().clone()
    attention_mask = attention_mask[:, None, None, :]  # Equivalent to jnp.newaxis
    attention_mask *= get_large_negative_number(dtype)
    return attention_mask


def causal_mask(input_t: torch.Tensor) -> torch.Tensor:
    """Computes and returns causal mask.

    Args:
        input_t: A torch.Tensor of shape [B, T, D].

    Returns:
        An attention_mask torch.Tensor of shape [1, 1, T, T]. Attention mask has
        already been converted to large negative values.
    """
    assert input_t.dtype.is_floating_point, input_t.dtype
    large_negative_number = get_large_negative_number(input_t.dtype)
    t = input_t.shape[1]
    col_idx = torch.arange(t).unsqueeze(0).repeat(t, 1)
    row_idx = torch.arange(t).unsqueeze(1).repeat(1, t)
    mask = (row_idx < col_idx).to(input_t.dtype) * large_negative_number
    return (mask.unsqueeze(0).unsqueeze(0).to(input_t.device)
            )  # Equivalent to jnp.newaxis


def merge_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Merges 2 masks.

    logscale mask is expected but 0/1 mask is also fine.

    Args:
        a: torch.Tensor of shape [1|B, 1, 1|T, S].
        b: torch.Tensor of shape [1|B, 1, 1|T, S].

    Returns:
        torch.Tensor of shape [1|B, 1, 1|T, S].
    """

    def expand_t(key_mask):
        query_mask = key_mask.transpose(-1, -2)  # Equivalent of jnp.transpose
        return torch.minimum(query_mask, key_mask)

    if a.shape[2] != b.shape[2]:
        if a.shape[2] == 1:
            a = expand_t(a)
        else:
            assert b.shape[2] == 1
            b = expand_t(b)

    assert a.shape[1:] == b.shape[1:], f"a.shape={a.shape}, b.shape={b.shape}."
    return torch.minimum(a, b)  # Element-wise minimum, similar to jnp.minimum

