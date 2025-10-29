import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext

from multiattention import MultiHeadAttention
from layernorm import FeedForward, LayerNorm

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
        每个EncoderLayer包括两个子层: 多头注意力层和前馈神经网络层。每个子层都使用了残差连接和层归一化。
        """
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        原文中使用: LayerNorm(x + SubLayer(x))
        也有部分实现使用: x + SubLayer(LayerNorm(x))
        这里我们使用原文的实现
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        output, _ = self.attn(x, x, x, mask=src_mask)
        x = self.norm_1(x + self.dropout_1(output))  # 多头自注意力子层
        x = self.norm_2(x + self.dropout_2(self.ff(x)))  # 前馈神经网络子层
        return x


# 编码器
class Encoder(nn.Module):
    """
    编码器由多个编码器层堆叠而成。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
        在原始论文的图 1 和描述中, 作者提到每个子层(Multi-Head Attention 和 Feed-Forward Network)之后会进行 Layer Normalization。
        但是，论文并没有明确提到在整个编码器或解码器之后进行额外的 Layer Normalization。
        许多后续的实现，通常会在编码器和解码器的堆叠之后再进行一次 Layer Normalization。
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        """
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
        每个DecoderLayer包括三个子层: 自注意力层、编码器-解码器注意力层和前馈神经网络层。每个子层都使用了残差连接和层归一化。
        """
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, memory_mask=None, tgt_mask=None):
        """
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        output_1, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm_1(x + self.dropout_1(output_1))  # 第一个子层：多头自注意力层

        output_2, _ = self.enc_dec_attn(x, enc_output, enc_output, mask=memory_mask)  # k, v来自编码器输出
        x = self.norm_2(x + self.dropout_2(output_2))  # 第二个子层：编码器-解码器注意力层

        x = self.norm_3(x + self.dropout_3(self.ff(x)))  # 第三个子层：前馈神经网络层
        return x


# 解码器
class Decoder(nn.Module):
    """
    解码器由多个解码器层堆叠而成。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, memory_mask=None, tgt_mask=None):
        """
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, enc_output, memory_mask, tgt_mask)
        return self.norm(x)
