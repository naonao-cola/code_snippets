from torchtext.data.metrics import (
    bleu_score,
)  # torchtext已停止更新，不影响使用，torchtext0.18版本对应PyTorch2.3版本，更高的pytorch版本导入torchtext将报错
import json
import opencc  # 使用的数据集是繁体字，用该库转换为简体字

from torchtext.data.utils import get_tokenizer  # 用于英文分词
import jieba  # 用于中文分词
from tqdm import tqdm
from collections import Counter  #  用于统计词频
import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# 项目根目录（transformer/）和数据目录的绝对路径，保证从任意工作目录导入时都能正确找到 data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cmn-eng")

# 创建 OpenCC 转换器实例
converter = opencc.OpenCC("t2s")  # 't2s'表示从繁体到简体

# 将该数据集保存为json格式
dataset = []
with open(os.path.join(DATA_DIR, "cmn.txt"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        line = line.split("\t")
        en = line[0]
        cn = line[1]
        cn = converter.convert(cn)
        dataset.append({"english": en, "chinese": cn})

# 将每个字典逐行写入 JSON 文件
with open(os.path.join(DATA_DIR, "dataset.json"), "w", encoding="utf-8") as json_file:
    for item in dataset:
        json.dump(item, json_file, ensure_ascii=False)
        json_file.write("\n")


# 英文分词示例
en_tokenizer = get_tokenizer("basic_english")
text = "Hello! How are you doing today?"
tokens = en_tokenizer(text)
print(tokens)


# 词语级分词【构造的词典很大，训练可能较慢】
# def cn_tokenizer(text):
#     return jieba.lcut(text)
# 字符级分词【构造的词典较小，但可能导致语义不连贯】
def cn_tokenizer(text):
    return list(text)


text = "你好！你今天好吗？"
tokens = cn_tokenizer(text)
print(tokens)

# ----------------------------------------------------------------
# 读取dataset
dataset = []
with open(os.path.join(DATA_DIR, "dataset.json"), "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

# ----------------------------------------------------------------
# 构建词典
en_max_len = 0
cn_max_len = 0
en_vocab = []
cn_vocab = []
for data in tqdm(dataset, desc="Building Vocabulary"):
    en_text = data["english"]
    cn_text = data["chinese"]

    en_tokens = en_tokenizer(en_text)
    cn_tokens = cn_tokenizer(cn_text)
    en_max_len = max(en_max_len, len(en_tokens))
    cn_max_len = max(cn_max_len, len(cn_tokens))

    en_vocab.extend(en_tokens)
    cn_vocab.extend(cn_tokens)

en_counter = dict(Counter(en_vocab))
cn_counter = dict(Counter(cn_vocab))

# 保存词频统计
with open(os.path.join(DATA_DIR, "en_counter.json"), "w", encoding="utf-8") as f:
    json.dump(en_counter, f, ensure_ascii=False, indent=4)
with open(os.path.join(DATA_DIR, "cn_counter.json"), "w", encoding="utf-8") as f:
    json.dump(cn_counter, f, ensure_ascii=False, indent=4)


# 为简单起见，将数据集所有的token都添加到词典中，不考虑词频和未知token
# 定义特殊字符
start_token = "<sos>"
end_token = "<eos>"
pad_token = "<pad>"

special_tokens = [start_token, end_token, pad_token]
en_vocab = special_tokens + list(en_counter.keys())
cn_vocab = special_tokens + list(cn_counter.keys())

# 构建词典
en_dict_token2id = {token: i for i, token in enumerate(en_vocab)}
en_dict_id2token = {i: token for i, token in enumerate(en_vocab)}
cn_dict_token2id = {token: i for i, token in enumerate(cn_vocab)}
cn_dict_id2token = {i: token for i, token in enumerate(cn_vocab)}

# 分别保存token到token_id和token_id到token的词典
with open(os.path.join(DATA_DIR, "en_dict_token2id.json"), "w") as f:
    json.dump(en_dict_token2id, f, ensure_ascii=False, indent=4)
with open(os.path.join(DATA_DIR, "cn_dict_token2id.json"), "w") as f:
    json.dump(cn_dict_token2id, f, ensure_ascii=False, indent=4)
with open(os.path.join(DATA_DIR, "en_dict_id2token.json"), "w") as f:
    json.dump(en_dict_id2token, f, ensure_ascii=False, indent=4)
with open(os.path.join(DATA_DIR, "cn_dict_id2token.json"), "w") as f:
    json.dump(cn_dict_id2token, f, ensure_ascii=False, indent=4)

# 计算词典大小
en_vocab_size = len(en_vocab)
cn_vocab_size = len(cn_vocab)
print(f"英文字典大小为：{en_vocab_size}")
print(f"英文最长序列长度为：{en_max_len}")
print(f"中文字典大小为：{cn_vocab_size}")
print(f"中文最长序列长度为：{cn_max_len}")


# 由token转换为token_id
def text2id(text, language, dict=None, dict_path=None, en_max_len=45, cn_max_len=50):
    """将一段文本转换为该词典下对应的token_id, 并根据max_len补全pad, 这里将中文填充为50, 英文填充为45

    Args:
        text : 输入文本
        language : 语言, 中文或英文
        dict : 词典, 如果为None, 则从dict_path中加载词典
        dict_path : 词典路径

    Returns:
        list : 列表, 里面是每个token的token_id
    """
    if language == "cn":
        token = cn_tokenizer(text)
        max_len = cn_max_len
    if language == "en":
        token = en_tokenizer(text)
        max_len = en_max_len
    if dict is None:
        if dict_path is None:
            dict_path = DATA_DIR
        with open(os.path.join(dict_path, f"{language}_dict_token2id.json"), "r") as f:
            dict = json.load(f)

    token_id = [dict[t] for t in token]
    token_id = [dict["<sos>"]] + token_id + [dict["<eos>"]]
    if len(token_id) < max_len:
        token_id += [dict["<pad>"]] * (max_len - len(token_id))
    return token_id


# 由token_id转换为token
def id2text(token_id, language, dict=None, dict_path=None):
    """将一个列表中的token_id转换为对应的文本, 并去掉<sos>、<eos>、<pad>

    Args:
        token_id : 装有token_id的列表
        language : 语言, 中文或英文
        dict : 词典, 如果为None, 则从dict_path中加载词典
        dict_path : 词典路径

    Returns:
        str : 文本
    """
    if dict is None:
        if dict_path is None:
            dict_path = DATA_DIR
        with open(os.path.join(dict_path, f"{language}_dict_id2token.json"), "r") as f:
            dict = json.load(f)
            dict = {int(k): v for k, v in dict.items()}  # 词典保存为json后，键会变成字符串, 转换为int

    token = [dict[i] for i in token_id if i not in [0, 1, 2]]
    if language == "cn":
        return "".join(token)
    if language == "en":
        text = ""
        CAP = False  # 调整是否大写
        # 调整英文单词、符号之间空格的有无
        for i, t in enumerate(token):
            if i == 0:
                text += t.capitalize()  # 首字母大写
            else:
                if t in ",.!?;:)}]'\"":
                    text += t
                    if t in ".?!":
                        CAP = True
                else:
                    if CAP:
                        t = t.capitalize()
                        CAP = False
                    text += " " + t
        return text


# 示例，由于未加入unk，必须使用词典里有的单词
# 中文token转token_id
cn_text = "你好吗？我很好！谢谢，不客气。"
cn_token_id = text2id(cn_text, "cn", cn_dict_token2id)
print(cn_token_id)  # <sos>、<eos>、<pad>的token_id分别为0、1、2
print("length:", len(cn_token_id))
# 英文token转token_id
en_text = "How are you? I am fine! Thank you, you are welcome."
en_token_id = text2id(en_text, "en", en_dict_token2id)
print(en_token_id)
print("length:", len(en_token_id))

# 中文token_id转token
print(id2text(cn_token_id, "cn", cn_dict_id2token))
# 英文token_id转token
print(id2text(en_token_id, "en", en_dict_id2token))


from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np


class TranslateDataset(Dataset):
    def __init__(self, dataset, en_dict_token2id, cn_dict_token2id):
        """构建中英翻译数据集

        Args:
            dataset : [{'english': '...', 'chinese': '...'}, ...]
            en_dict_token2id : 英语字典, token到token_id的映射
            cn_dict_token2id : 中文字典, token到token_id的映射
        """
        self.dataset = dataset
        self.en_token_id_data = []
        self.cn_token_id_data = []
        for data in self.dataset:
            self.en_token_id_data.append(text2id(data["english"], language="en", dict=en_dict_token2id))
            self.cn_token_id_data.append(text2id(data["chinese"], language="cn", dict=cn_dict_token2id))
        self.en_token_id_data = np.array(self.en_token_id_data)  # (total_data_size, en_seq_len)
        self.cn_token_id_data = np.array(self.cn_token_id_data)  # (total_data_size, cn_seq_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.en_token_id_data[index, :], self.cn_token_id_data[index, :]


# 构建数据集
translate_dataset = TranslateDataset(dataset, en_dict_token2id, cn_dict_token2id)
# 划分数据集
val_size = 1000  # 只选用1000个数据验证
train_size = len(dataset) - val_size
# 分割数据集
train_dataset, val_dataset = random_split(translate_dataset, [train_size, val_size])
# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

for en_token_id, cn_token_id in train_loader:
    print(en_token_id)
    print(cn_token_id)
    break

from model import *
from mask import *

# 构建transformer模型
# 由于数据集较小，这里我们构建一个小型transformer
transformer = Transformer(
    src_vocab_size=en_vocab_size,  # 7192
    tgt_vocab_size=cn_vocab_size,  # 2839
    d_model=256,
    num_layers=3,
    num_heads=8,
    d_ff=1024,
    dropout=0.1,
    max_len=100,  # max_len可以取大一些，实际计算时只会根据序列长度取相应尺度的位置编码
)

test_src = torch.randint(0, en_vocab_size, (32, 45))  # 在词典范围内生成随机数
test_tgt = torch.randint(0, cn_vocab_size, (32, 50))
test_out = transformer(test_src, test_tgt)
print(test_out.shape)

trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {trainable_params}")


# 定义参数
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = transformer.to(device)

loss_func = torch.nn.CrossEntropyLoss(ignore_index=2)  # 计算损失时，忽略掉pad_id部分的计算
optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.8
)  # 每隔固定数量的epoch将学习率减少一个固定的比例

train_loss_curve = []
val_loss_curve = []
lr_curve = []
# 训练和验证
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    transformer.train()
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

        # 构造mask
        src_mask = make_src_mask(src=src, pad_id=2)
        memory_mask = make_memory_mask(src=src, pad_id=2)
        tgt_mask = make_tgt_mask(tgt=tgt[:, :-1], pad_id=2)

        ####################################################
        # mask可视化
        if epoch == 0 and step == 0:
            # print(src_mask.shape)
            # print(src_mask)
            # print(memory_mask.shape)
            # print(memory_mask)
            # print(tgt_mask.shape)
            # print(tgt_mask)
            plt.imshow(
                src_mask.squeeze().numpy(), cmap="viridis", interpolation="nearest"
            )  # (batch_size, seq_len)
            plt.colorbar()
            plt.title("src_mask")
            plt.show()
            plt.imshow(
                memory_mask.squeeze().numpy(), cmap="viridis", interpolation="nearest"
            )  # (batch_size, seq_len)
            plt.colorbar()
            plt.title("memory_mask")
            plt.show()
            plt.imshow(
                tgt_mask[0].squeeze().numpy(), cmap="viridis", interpolation="nearest"
            )  # 取了batch中的第一个，(seq_len, seq_len)
            plt.colorbar()
            plt.title("tgt_mask")
            plt.show()
        ####################################################

        src = src.to(device).long()
        tgt = tgt.to(device).long()
        src_mask = src_mask.to(device)
        memory_mask = memory_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        # 训练时，是由输入的tgt预测下一个字符，因此输入为tgt[:, :-1]，每一个位置的字符在看见前面已有字符的情况下预测下一个字符
        # 例如，tgt为: <sos> a b c d e <eos> <pad> <pad>，那么输入为：<sos> a b c d e <eos> <pad>，真值为：a b c d e <eos> <pad> <pad>
        # 假设预测输出为：a' b' c' d' e' <eos> <pad> <pad>，该预测输出需要与真实值进行交叉熵计算损失，为避免<pad>对有效token的影响，计算损失时<pad>位置不参与
        # 因此实际需要计算的是：a b c d e <eos> 与 a' b' c' d' e' <eos>的对应字符位置损失
        pred = transformer(src, tgt[:, :-1], src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)

        ####################################################
        # 查看训练时翻译效果
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
    transformer.eval()
    loss_sum = 0.0
    for step, (src, tgt) in enumerate(val_loader):
        # 构造mask
        src_mask = make_src_mask(src=src, pad_id=2)
        memory_mask = make_memory_mask(src=src, pad_id=2)
        tgt_mask = make_tgt_mask(tgt=tgt[:, :-1], pad_id=2)

        src = src.to(device).long()
        tgt = tgt.to(device).long()
        src_mask = src_mask.to(device)
        memory_mask = memory_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        pred = transformer(src, tgt[:, :-1], src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)
        pred = pred.contiguous().view(-1, pred.shape[-1])
        target = tgt[:, 1:].contiguous().view(-1)
        loss = loss_func(pred, target)

        loss_sum += loss.item()

    val_avg_loss = loss_sum / len(val_loader)
    val_loss_curve.append(val_avg_loss)
    print(f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | LR: {lr:.6f}")


# 保存模型
torch.save(transformer.state_dict(), "transformer_from_scratch.pt")

# 绘制损失曲线
plt.figure()
plt.plot(train_loss_curve, label="Train Loss", color="blue")
plt.plot(val_loss_curve, label="Validation Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.show()
plt.savefig("loss_curve.png")

# 绘制学习率曲线
plt.figure()
plt.plot(lr_curve, label="Learning Rate", color="green")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Curve")
plt.legend()
plt.show()
plt.savefig("lr_curve.png")
