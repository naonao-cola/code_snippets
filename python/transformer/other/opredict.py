import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from odemo import en_dict_token2id, cn_dict_token2id, cn_dict_id2token, torch_transformer, text2id, id2text
from mask import make_padding_mask, make_sequence_mask

# 没有条件训练可以跳过上一个单元格，加载训练好的模型:transformer_from_scratch.pt
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_transformer.load_state_dict(torch.load("transformer_from_torch.pt", map_location=device))
torch_transformer = torch_transformer.to(device)


# 构造一个用于预测的函数
def predict_torch(src):
    """
    接收一个源序列，根据源序列，从<sos>开始生成目标序列
    src: (batch_size, src_seq_len)
    """
    torch_transformer.eval()

    # 初始化tgt，从<sos>开始，后面全部填充为<pad>
    batch_size = src.size(0)  # 获取batch_size
    tgt = [cn_dict_token2id["<sos>"]] + [
        cn_dict_token2id["<pad>"]
    ] * 48  # 目表序列填充的长度是50，训练时使用49，因此预测时初始化为49
    tgt = torch.LongTensor(tgt).unsqueeze(0)  # (1, tgt_seq_len)
    tgt = tgt.repeat(batch_size, 1)  # (batch_size, tgt_seq_len)

    # 构造mask 【这里只构造src_pad_mask，也就是第二部分的src_mask，实际上src_mask和memory_mask是构造时是相同的，只是内部实现时广播后的维度有区别】
    src_pad_mask = make_padding_mask(src, pad_id=2, return_int=False, true_to_mask=True)

    # 选择设备
    src = src.to(device).long()
    src_pad_mask = src_pad_mask.to(device)

    # 将src的编码器输出存为变量，解码器计算时可以重复使用
    src = torch_transformer.positional_encoding(torch_transformer.src_embedding(src))
    # 【这里需使用nn.Transformer类内定义的encoder，后面有涉及到的也是一样，传入参数需参考官方文档，这里我们只用为关键字src和src_key_padding_mask传入值】
    memory = torch_transformer.transformer.encoder(src=src, src_key_padding_mask=src_pad_mask)

    for i in range(48):  # 逐字符生成
        tgt_temp = tgt  # 赋值给tgt_temp，充当解码器输入，预测的输出更新tgt，而后再返回充当新一轮的解码器输入

        # 构造mask 【这里构造tgt的mask，分别为tgt_pad_mask和tgt_seq_mask，二者会在模型内部合并。在第二部分中，我们是在模型外通过定义函数合并的】
        tgt_pad_mask = make_padding_mask(tgt_temp, pad_id=2, return_int=False, true_to_mask=True)
        tgt_seq_mask = make_sequence_mask(tgt_temp, return_int=False, true_to_mask=True)
        tgt_temp = tgt_temp.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        tgt_seq_mask = tgt_seq_mask.to(device)

        tgt_temp = torch_transformer.positional_encoding(torch_transformer.tgt_embedding(tgt_temp))
        tgt_temp = torch_transformer.transformer.decoder(
            tgt=tgt_temp,
            memory=memory,
            tgt_mask=tgt_seq_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        out = torch_transformer.fc_out(tgt_temp)  # (batch, tgt_seq_len, tgt_vocab_size)

        # 当前预测是第i个词，故取出第i个
        # 第一步输入为：<sos> <pad> <pad> <pad> <pad> <pad> <pad>
        # 第一步预测为：a <pad>' <pad>' <pad>' <pad>' <pad>' <pad>'  (由于<pad>不参与损失计算，预测输出<pad>'实际没有上是没有意义的输出，并非实际的<pad>字符)
        # 取出第一步预测的a，构建第二步输入：<sos> a <pad> <pad> <pad> <pad> <pad>
        # 第二步预测为：a b <pad>' <pad>' <pad>' <pad>' <pad>'，取出第二步预测的b，构建第三步输入：<sos> a b <pad> <pad> <pad> <pad>，直到预测到<eos>结束
        out = out[:, i, :]  # (batch, tgt_seq_len, tgt_vocab_size) -> (batch, tgt_vocab_len)
        # 将预测的logits映射到具体的toekn_id
        out = out.argmax(dim=1).detach()  # 在tgt_vocab_size维度上取最大值，得到预测的token_id -> (batch,)

        # 将本轮预测的词加入到tgt中，用于下一轮预测
        tgt[:, i + 1] = out
        # 如果预测的out为<eos>，说明预测结束，返回tgt
        # 本函数仅用于单个字符串预测，因此检查一个序列是否产生<eos>
        # 如果预测多个序列，需要添加逻辑用于跟踪所有序列是否均产生<eos>再退出
        if out == 1:
            return tgt

    # 如果未能预测到<eos>，循环结束直接返回tgt
    return tgt


# 实测
english_text = "This is the last part."

# 将文本转换成token_id
english_token_id = text2id(english_text, "en", en_dict_token2id)
print("English token id:", english_token_id)

# 预测
src = torch.tensor(english_token_id).unsqueeze(0)  # (1, 45)
predict_token_id = predict_torch(src).squeeze(0).tolist()
print("Predict token id:", predict_token_id)

# 将预测输出的token_id转换成文本
chinese_text = id2text(predict_token_id, "cn", cn_dict_id2token)
print("Predict chinese text:", chinese_text)
