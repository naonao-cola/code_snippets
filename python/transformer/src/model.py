import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext

from multiattention import MultiHeadAttention
from layernorm import FeedForward, LayerNorm
from coder import Encoder, Decoder
from position import PositionalEncoding


# 完整Transformer模型
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=500,
    ):
        super().__init__()
        # src_vocab_size和tgt_vocab_size分别是源序列和目标序列的词典大小
        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model
        )  # 定义嵌入层，用于将序列转换为维度为d_model的嵌入向量
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)  # 位置编码层

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        src和tgt为token_id
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        在 Transformer 模型中, 输入序列通常已经经过填充(padding)处理。
        填充是为了使所有输入序列的长度一致，从而可以将它们放入一个批次中进行处理。
        """
        src = self.dropout(
            self.positional_encoding(self.src_embedding(src))
        )  # 位置编码后使用了dropout，原文在Regularization中有提到
        tgt = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, memory_mask, tgt_mask)

        # 在训练过程中，logits 通常会通过 CrossEntropyLoss 来计算损失，而 CrossEntropyLoss 会在内部应用 softmax
        # 因此这里可以不用softmax，在推理阶段，可以在output后手动加入softmax
        output = self.fc_out(dec_output)
        return output
