import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext
from demo import en_dict_token2id, cn_dict_token2id, cn_dict_id2token, transformer,text2id, id2text
from mask import make_src_mask, make_tgt_mask, make_memory_mask

# 没有条件训练可以跳过上一个单元格，加载训练好的模型:transformer_from_scratch.pt
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer.load_state_dict(torch.load("./transformer_from_scratch.pt", map_location=device))
transformer = transformer.to(device)


# 构造一个用于预测的函数
def predict(src):
    """
    接收一个源序列，根据源序列，从<sos>开始生成目标序列
    src: (batch_size, src_seq_len)
    """
    transformer.eval()

    # 初始化tgt，从<sos>开始，后面全部填充为<pad>
    batch_size = src.size(0)  # 获取batch_size
    tgt = [cn_dict_token2id["<sos>"]] + [
        cn_dict_token2id["<pad>"]
    ] * 48  # 目表序列填充的长度是50，训练时使用49，因此预测时初始化为49
    tgt = torch.LongTensor(tgt).unsqueeze(0)  # (1, tgt_seq_len)
    tgt = tgt.repeat(batch_size, 1)  # (batch_size, tgt_seq_len)

    # 构造mask
    src_mask = make_src_mask(src=src, pad_id=2)
    memory_mask = make_memory_mask(src=src, pad_id=2)

    # 选择设备
    src = src.to(device).long()
    src_mask = src_mask.to(device)
    memory_mask = memory_mask.to(device)

    # 将src的编码器输出存为变量，解码器计算时可以重复使用
    src = transformer.positional_encoding(transformer.src_embedding(src))
    memory = transformer.encoder(src, src_mask=src_mask)

    for i in range(48):  # 逐字符生成
        tgt_temp = tgt  # 赋值给tgt_temp，充当解码器输入，预测的输出更新tgt，而后再返回充当新一轮的解码器输入

        # 构造mask
        tgt_mask = make_tgt_mask(tgt=tgt_temp, pad_id=2)
        tgt_temp = tgt_temp.to(device)
        tgt_mask = tgt_mask.to(device)

        tgt_temp = transformer.positional_encoding(transformer.tgt_embedding(tgt_temp))
        tgt_temp = transformer.decoder(
            tgt_temp, enc_output=memory, memory_mask=memory_mask, tgt_mask=tgt_mask
        )
        out = transformer.fc_out(tgt_temp)  # (batch, tgt_seq_len, tgt_vocab_size)

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
english_text = "This is the first part."

# 将文本转换成token_id
english_token_id = text2id(english_text, "en", en_dict_token2id)
print("English token id:", english_token_id)

# 预测
src = torch.tensor(english_token_id).unsqueeze(0)  # (1, 45)
predict_token_id = predict(src).squeeze(0).tolist()
print("Predict token id:", predict_token_id)

# 将预测输出的token_id转换成文本
chinese_text = id2text(predict_token_id, "cn", cn_dict_id2token)
print("Predict chinese text:", chinese_text)
