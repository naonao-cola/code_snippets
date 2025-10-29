import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext


# 填充掩码
def make_padding_mask(seq, pad_id, return_int=True, true_to_mask=False):
    """
    构造padding mask, 参数设置根据不同的Transformer实现来确定
    Args:
        seq: 需要构造mask的序列(batch, seq_len), 该序列使还未进行Embedding, 里面放的是token_id
        pad_id: 用于填充的特殊字符<PAD>所对应的token_id, 根据不同代码设置
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        mask: (batch, seq_len), 不同的Transformer实现需输入的形状也不同, 根据需要进行后续更改
    """
    mask = seq == pad_id  # (batch, seq_len), 在<PAD>的位置上生成True, 真实序列的位置为False

    if true_to_mask is False:
        mask = ~mask

    if return_int:
        mask = mask.int()

    return mask


# 因果掩码
def make_sequence_mask(seq, return_int=True, true_to_mask=False):
    """
    构造sequence mask, 参数设置根据不同的Transformer实现来确定
    Args:
        seq: 需要构造mask的序列(batch, seq_len), 该序列使还未进行Embedding, 里面放的是token_id
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        mask: (seq_len, seq_len), 不同的Transformer实现需输入的形状也不同, 根据需要进行后续更改
    """
    _, seq_len = seq.shape
    mask = torch.tril(torch.ones(seq_len, seq_len))  # (seq_len, seq_len), 下三角为1, 上三角为0
    mask = 1 - mask
    mask = mask.bool()

    if true_to_mask is False:
        mask = ~mask

    if return_int:
        mask = mask.int()

    return mask


# 示例
seq = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
)  # (batch_size, src_seq_len=12)

# 比较四种不同的padding_mask
print(make_padding_mask(seq=seq, pad_id=0, return_int=True, true_to_mask=False))
print(make_padding_mask(seq=seq, pad_id=0, return_int=True, true_to_mask=True))
print(make_padding_mask(seq=seq, pad_id=0, return_int=False, true_to_mask=False))
print(make_padding_mask(seq=seq, pad_id=0, return_int=False, true_to_mask=True))

# 展示sequence_mask
print(make_sequence_mask(seq=seq, return_int=True, true_to_mask=False))


# 进一步分别构造src_mask、memory_mask、tgt_mask
def make_src_mask(src, pad_id, return_int=True, true_to_mask=False):
    """构造src_mask

    Args:
        src: 源序列(batch_size, src_len)
        pad_id: 补全符号的token_id
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        src_mask: (batch_size, 1, 1, src_len)
    """
    padding_mask = make_padding_mask(src, pad_id, return_int=return_int, true_to_mask=true_to_mask)
    padding_mask = padding_mask.unsqueeze(1)
    padding_mask = padding_mask.unsqueeze(2)
    return padding_mask


def make_memory_mask(src, pad_id, return_int=True, true_to_mask=False):
    """构造memory_mask

    Args:
        src: 源序列(batch_size, src_len)
        pad_id: 补全符号的token_id
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        memory_mask: (batch_size, 1, 1, src_len)
    """
    padding_mask = make_padding_mask(src, pad_id, return_int=return_int, true_to_mask=true_to_mask)
    padding_mask = padding_mask.unsqueeze(1)
    padding_mask = padding_mask.unsqueeze(2)
    return padding_mask


def make_tgt_mask(tgt, pad_id, return_int=True, true_to_mask=False):
    """构造tgt_mask

    Args:
        tgt: 目标序列(batch_size, tgt_len)
        pad_id: 补全符号的token_id
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        tgt_mask: (batch_size, 1, tgt_len, tgt_len)
    """
    padding_mask = make_padding_mask(
        tgt, pad_id, return_int=return_int, true_to_mask=true_to_mask
    )  # (batch_size, tgt_len)
    padding_mask = padding_mask.unsqueeze(1)
    padding_mask = padding_mask.unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
    padding_mask = padding_mask.repeat(1, 1, tgt.size(1), 1)  # (batch_size, 1, tgt_len, tgt_len)

    sequence_mask = make_sequence_mask(tgt, return_int=True, true_to_mask=False)  # (tgt_len, tgt_len)
    sequence_mask = sequence_mask.unsqueeze(0)
    sequence_mask = sequence_mask.unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
    sequence_mask = sequence_mask.repeat(tgt.size(0), 1, 1, 1)  # (batch_size, 1, tgt_len, tgt_len)

    # 合并两个mask
    if true_to_mask is False:  # 根据不同类型的mask, 使用"与"或"或"的方式进行合并
        mask = padding_mask & sequence_mask
    else:
        mask = padding_mask | sequence_mask
    return mask


# 示例
src = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
tgt = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])

src_mask = make_src_mask(src, pad_id=0)
print("src_mask:")
print(src_mask)
print(src_mask.shape)

memory_mask = make_memory_mask(
    src, pad_id=0
)  # memory_mask和src_mask这里是一样的, 但在transformer内部会广播成不同的维度
print("memory_mask:")
print(memory_mask)
print(memory_mask.shape)

tgt_mask = make_tgt_mask(tgt, pad_id=0)
print("tgt_mask:")
print(tgt_mask)
print(tgt_mask.shape)
