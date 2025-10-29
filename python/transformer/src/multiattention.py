import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext


class MultiHeadAttention(nn.Module):
    """ 多头自注意力机制
    """
    def __init__(self, d_model,num_heads,dropout=0.1):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"  # 确保num_heads能整除d_model
        self.d_model = d_model
        self.d_k = (
            d_model // num_heads
        )  # 每个头的维度,# 这里简单起见，我们只考虑 d_v = d_k = d_q = d_model / num_heads，因此只定义d_k

        self.h = num_heads  # 头的数量

        # 这里定义的 linear 参数是 (d_model, d_model)
        self.q_linear = nn.Linear(d_model, d_model)  # W_Q
        self.v_linear = nn.Linear(d_model, d_model)  # W_K
        self.k_linear = nn.Linear(d_model, d_model)  # W_V
        self.o_linear = nn.Linear(d_model, d_model)  # W_O
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        input:
            q, k, v: (batch_size, seq_len, d_model)
                对于自注意力, 如果输入序列为 x, 那么 q=x, k=x, v=x
                对于交叉注意力, 如果序列 x_1 对序列 x_2 做 query, 则 q=x_1, k=x_2, v=x_2
            mask: (batch_size, 1, 1, seq_len)或(batch_size, 1, seq_len, seq_len)
                mask有多种形式, 可以使用0、1来mask, 也可以使用True、False来mask, 根据具体代码执行mask
        output:
            seq: (batch_size, seq_len, d_model)
            attention: (batch_size, h, len_q, len_k) 每个头均有一个注意力权重矩阵
                对于自注意力, len_q = len_k = len_v = seq_len
                对于交叉注意力, len_q = tgt_seq_len , len_k = len_v = src_seq_len
        """
        batch_size = q.size(0)

        # 将原始序列变换为QKV矩阵
        # 以 q 的变换为例。序列 q=x 经过 q_linear 变换后，形状仍然为(batch_size, seq_len, d_model)
        # 使用.view方法用于改变张量形状。这里变换成了(batch_size, seq_len, num_heads, d_k)，即把 d_model 拆成了 num_heads*d_k
        # 使用.transpose方法，将形状进一步变为(batch_size, num_heads, seq_len, d_k)
        q = (
            self.q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch_size, seq_len, d_model)->(batch_size, num_heads, seq_len, d_k)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 每个头并行计算相似度得分，相似度矩阵形状为(batch_size, num_heads, len_q, len_k)
        # 即每个头都形成了(len_q, len_k)的 scores，scores 的第一行，意思是第一个位置的 q 对所有位置的 k 的得分，因此后续的 softmax 是按 scores 的行来做的
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 这里我们假设mask中为0的地方是需要遮蔽的地方
            scores = scores.masked_fill(
                mask == 0, -1e9
            )  # 通过把掩码的位置设置为一个较大的负数，让掩码位置的softmax趋近于零

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)  # 得到所有batch的每个头的相似度矩阵

        # 相似度矩阵与v相乘得到输出
        output = torch.matmul(attention, v)  # (batch_size, num_heads, seq_len, d_k)

        # 首先将output变为(batch_size, seq_len, num_heads, d_k)
        # .contiguous用于确保张量在内存中是连续的
        # 将张量形状变为(batch_size, seq_len, d_model)，相当于把所有头的结果拼接了起来，即 d_k*num_heads 拼成了 d_model
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o_linear(output)  # 使用w_o进行线性变换

        # 最终传出输出和每个头的attention，attention根据需要可用于后续的可视化
        return output, attention

# 示例用法
d_model = 512
mha = MultiHeadAttention(d_model=512, num_heads=8)

x = torch.randn(32, 10, d_model)  # 长度为10的序列矩阵

output, attention = mha(x, x, x)

print(output.shape)
print(attention.shape)

# 可视化第0个batch的第0个头的attention
head_attention = attention[0, 0].detach().numpy()  # 提取第 0 个 batch 的第 0 个头

# 绘制热力图
plt.imshow(head_attention, cmap="viridis")
plt.colorbar()
plt.title("Attention Weights")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.show()
plt.savefig("attention_weights.png")
