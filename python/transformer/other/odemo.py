import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from position import PositionalEncoding
from demo import en_vocab_size, cn_vocab_size
from demo import id2text, text2id, train_loader, en_dict_id2token, cn_dict_id2token, val_loader
from tqdm import tqdm
from mask import make_padding_mask, make_sequence_mask

# nn.Transformer没有实现Embedding,PositionalEncoding和最后的Linear, 因此需要自己封装
class TorchTransformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        dropout=0.1,
    ):
        super(TorchTransformer, self).__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src, tgt, src_pad_mask, tgt_pad_mask, tgt_seq_mask
    ):  # tgt的padding mask和sequence mask可以分别传入，模型内会进行合并
        src = self.dropout(
            self.positional_encoding(self.src_embedding(src))
        )  # (batch,src_seq_len)->(batch,src_seq_len,d_model)
        tgt = self.dropout(
            self.positional_encoding(self.tgt_embedding(tgt))
        )  # (batch,tgt_seq_len)->(batch,tgt_seq_len,d_model)
        # 编码器中的多头注意力使用src_key_padding_mask，传入的是src_pad_mask
        # 解码器中的多头注意力使用tgt_key_padding_mask和tgt_mask，传入的是tgt_pad_mask和tgt_seq_mask，二者会进行合并
        # 解码器中的交叉注意力使用memory_key_padding_mask，传入的是src_pad_mask
        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_seq_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        output = self.fc_out(output)
        return output


# 整体流程与第二部分相同，不同的是传入的mask的部分，mask需要传入bool形式，True表示遮蔽
# 运行此部分代码需要第二部分的（一）至（四）的代码块已经运行过，部分变量已经在内存中

# 构建模型
torch_transformer = TorchTransformer(
    src_vocab_size=en_vocab_size,
    tgt_vocab_size=cn_vocab_size,
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    max_seq_length=100,
    dropout=0.1,
)

trainable_params = sum(p.numel() for p in torch_transformer.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {trainable_params}")

import torch.nn.init as init


# nn.Transformer默认的初始化方式是Xavier，在此数据集下，我们定义的学习率调度和迭代次数训练效果并不理想
# 我们这里将初始化方式改为pytorch默认的方式，能够得到较好的效果，这是个有趣的现象
def reset_to_default_init(module):
    if isinstance(module, nn.Linear):
        # PyTorch 默认初始化方式：Kaiming 均匀分布
        init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            # PyTorch 默认的偏置初始化
            fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(module.bias, -bound, bound)


torch_transformer.apply(reset_to_default_init)


# 定义参数
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_transformer = torch_transformer.to(device)

loss_func = torch.nn.CrossEntropyLoss(ignore_index=2)  # 计算损失时，忽略掉pad_id部分的计算
optimizer = torch.optim.AdamW(torch_transformer.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.8
)  # 每隔固定数量的epoch将学习率减少一个固定的比例

train_loss_curve = []
val_loss_curve = []
lr_curve = []
# 训练和验证
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    torch_transformer.train()
    loss_sum = 0.0

    # 训练----------------------------------------------------
    for step, (src, tgt) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # src: (batch_size, 45)
        # tgt: (batch_size, 50)

        ####################################################
        if step % (len(train_loader) - 1) == 0 and step != 0:
            print(id2text(src[0].tolist(), "en", en_dict_id2token))
            print(id2text(tgt[0].tolist(), "cn", cn_dict_id2token))
        ####################################################

        # 构造mask 【此处构造适合nn.Transformer的mask】
        src_pad_mask = make_padding_mask(
            src, pad_id=2, return_int=False, true_to_mask=True
        )  # (batch_size, seq_len)
        tgt_pad_mask = make_padding_mask(
            tgt[:, :-1], pad_id=2, return_int=False, true_to_mask=True
        )  # (batch_size, seq_len)
        tgt_seq_mask = make_sequence_mask(
            tgt[:, :-1], return_int=False, true_to_mask=True
        )  # 需传入(T, T)或(N*num_heads, T, T)的形状，这里我们就传入(T, T)，即(seq_len, seq_len)

        ####################################################
        if epoch == 0 and step == 0:
            # print(src_pad_mask.shape)
            # print(src_pad_mask)
            # print(tgt_pad_mask.shape)
            # print(tgt_pad_mask)
            # print(tgt_seq_mask.shape)
            # print(tgt_seq_mask)
            plt.imshow(src_pad_mask.numpy(), cmap="viridis", interpolation="nearest")
            plt.colorbar()  # 添加颜色条
            plt.title("src_pad_mask")
            plt.show()
            plt.imshow(tgt_pad_mask.numpy(), cmap="viridis", interpolation="nearest")
            plt.colorbar()  # 添加颜色条
            plt.title("tgt_pad_mask")
            plt.show()
            plt.imshow(tgt_seq_mask.numpy(), cmap="viridis", interpolation="nearest")
            plt.colorbar()  # 添加颜色条
            plt.title("tgt_seq_mask")
            plt.show()
        ####################################################
        src = src.to(device).long()
        tgt = tgt.to(device).long()
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        tgt_seq_mask = tgt_seq_mask.to(device)

        # 训练时，是由输入的tgt预测下一个字符，因此输入为tgt[:, :-1]，每一个位置的字符在看见前面已有字符的情况下预测下一个字符
        # 例如，tgt为: <sos> a b c d e <eos> <pad> <pad>，那么输入为：<sos> a b c d e <eos> <pad>，真值为：a b c d e <eos> <pad> <pad>
        # 假设预测输出为：a' b' c' d' e' <eos> <pad> <pad>，该预测输出需要与真实值进行交叉熵计算损失，为避免<pad>对有效token的影响，计算损失时<pad>位置不参与
        # 因此实际需要计算的是：a b c d e <eos> 与 a' b' c' d' e' <eos>的对应字符位置损失
        pred = torch_transformer(src, tgt[:, :-1], src_pad_mask, tgt_pad_mask, tgt_seq_mask)

        ####################################################
        if step % (len(train_loader) - 1) == 0 and step != 0:
            test_pred = pred[0]  # (seq_len, vocab_size)
            test_pred = test_pred.argmax(dim=1)  # (seq_len,)
            test_pred = test_pred.tolist()  # 转换成装了token_id的列表
            if 1 in test_pred:
                eos_index = test_pred.index(1)  # 找到<eos>索引
                test_pred = test_pred[: eos_index + 1]
            print(id2text(test_pred, "cn", cn_dict_id2token))
            print("pred_len:", len(test_pred))
        ####################################################

        # 调整形状以计算损失
        pred = pred.contiguous().view(
            -1, pred.shape[-1]
        )  # (batch_size, seq_len, cn_vocab_size) -> (batch_size * seq_len, cn_vocab_size)
        target = tgt[:, 1:].contiguous().view(-1)  # (batch_size, seq_len) -> (batch_size * seq_len)
        loss = loss_func(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()  # 当前epoch的累计损失

    train_avg_loss = loss_sum / len(train_loader)
    lr = optimizer.param_groups[0]["lr"]
    train_loss_curve.append(train_avg_loss)
    lr_curve.append(lr)

    scheduler.step()

    # 验证----------------------------------------------------
    torch_transformer.eval()
    loss_sum = 0.0
    for step, (src, tgt) in enumerate(val_loader):
        # 构造mask 【此处构造适合nn.Transformer的mask】
        src_pad_mask = make_padding_mask(
            src, pad_id=2, return_int=False, true_to_mask=True
        )  # (batch_size, seq_len)
        tgt_pad_mask = make_padding_mask(
            tgt[:, :-1], pad_id=2, return_int=False, true_to_mask=True
        )  # (batch_size, seq_len)
        tgt_seq_mask = make_sequence_mask(
            tgt[:, :-1], return_int=False, true_to_mask=True
        )  # 需传入(T, T)或(N*num_heads, T, T)的形状，这里我们就传入(T, T)，即(seq_len, seq_len)

        src = src.to(device).long()
        tgt = tgt.to(device).long()
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        tgt_seq_mask = tgt_seq_mask.to(device)

        pred = torch_transformer(src, tgt[:, :-1], src_pad_mask, tgt_pad_mask, tgt_seq_mask)
        pred = pred.contiguous().view(-1, pred.shape[-1])
        target = tgt[:, 1:].contiguous().view(-1)
        loss = loss_func(pred, target)

        loss_sum += loss.item()

    val_avg_loss = loss_sum / len(val_loader)
    val_loss_curve.append(val_avg_loss)
    print(f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | LR: {lr:.6f}")


# 保存模型
torch.save(torch_transformer.state_dict(), "transformer_from_torch.pt")

# 绘制损失曲线
plt.figure()
plt.plot(train_loss_curve, label="Train Loss", color="blue")
plt.plot(val_loss_curve, label="Validation Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.show()

# 绘制学习率曲线
plt.figure()
plt.plot(lr_curve, label="Learning Rate", color="green")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Curve")
plt.legend()
plt.show()
