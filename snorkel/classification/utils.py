from typing import Dict, List, Tuple, Union

import numpy as np
import torch

TensorCollection = Union[torch.Tensor, dict, list, tuple]


def list_to_tensor(item_list: List[torch.Tensor]) -> torch.Tensor:
    """Convert a list of torch.Tensor into a single torch.Tensor."""

    # Convert single value tensor
    if all(item_list[i].dim() == 0 for i in range(len(item_list))):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert 2 or more-D tensor with the same shape
    elif all(
        (item_list[i].size() == item_list[0].size()) and (len(item_list[i].size()) != 1)
        for i in range(len(item_list))
    ):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert reshape to 1-D tensor and then convert
    else:
        item_tensor, _ = pad_batch([item.view(-1) for item in item_list])

    return item_tensor


def pad_batch(
    batch: List[torch.Tensor],
    max_len: int = 0,
    pad_value: int = 0,
    left_padded: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert the batch into a padded tensor and mask tensor.

    Parameters
    ----------
    batch
        The data for padding
    max_len
        Max length of sequence of padding
    pad_value
        The value to use for padding
    left_padded
        If True, pad on the left, otherwise on the right

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The padded matrix and correspoing mask matrix.
    """

    batch_size = len(batch)
    max_seq_len = int(np.max([len(item) for item in batch]))  # type: ignore

    if max_len > 0 and max_len < max_seq_len:
        max_seq_len = max_len

    padded_batch = batch[0].new_full((batch_size, max_seq_len), pad_value)

    for i, item in enumerate(batch):
        length = min(len(item), max_seq_len)  # type: ignore
        if left_padded:
            padded_batch[i, -length:] = item[-length:]
        else:
            padded_batch[i, :length] = item[:length]

    mask_batch = torch.eq(padded_batch.clone().detach(), pad_value).type_as(
        padded_batch
    )

    return padded_batch, mask_batch


def get_active_mask(Y: torch.Tensor) -> torch.Tensor:
    """Return a mask of the active samples given some label Tensor Y.

    A sample is "active" if its corresponding label is not an abstain (with label -1).

    Parameters
    ----------
    Y
        Dimension N labels, where -1 indicates an abstain

    Returns
    -------
    Tensor
        Dimension N mask indicating "active" samples

    Example
    -------
    >>> Y = torch.LongTensor([-1, -1, 0, 1, 1])
    >>> get_active_mask(Y)
    tensor([0, 0, 1, 1, 1], dtype=torch.uint8)

    """
    if len(Y.size()) == 1:
        active = Y.detach() != -1
    else:
        active = torch.any(Y.detach() != -1, dim=1)
    return active


def move_to_device(
    obj: TensorCollection, device: int = -1
) -> TensorCollection:  # pragma: no cover
    """Recursively move torch.Tensors to a given CUDA device.

    Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
    to the specified GPU (or do nothing, if they should beon the CPU).

    Originally from:
    https://github.com/HazyResearch/metal/blob/mmtl_clean/metal/utils.py

    Parameters
    ----------
    obj
        Tensor or collection of Tensors to move
    device
        Device to move Tensors to
        device = -1 -> "cpu"
        device =  0 -> "cuda:0"
    """

    if device < 0 or not torch.cuda.is_available():
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(device)  # type: ignore
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def collect_flow_outputs_by_suffix(
    flow_dict: Dict[str, List[torch.Tensor]], suffix: str
) -> List[torch.Tensor]:
    """Return flow_dict outputs specified by suffix, ordered by sorted flow_name."""
    return [
        flow_dict[flow_name][0]
        for flow_name in sorted(flow_dict.keys())
        if flow_name.endswith(suffix)
    ]
