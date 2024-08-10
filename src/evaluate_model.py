import torch
import numpy as np
from tqdm import tqdm
from torch.cuda import device
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM



class Protein_dataset(Dataset):
    def __init__(self, lines: list[str], tokenizer: Tokenizer):
        self.lines = lines  # 存储数据集中的序列
        self.tokenizer = tokenizer  # 用于对序列进行编码的 Tokenizer 对象

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]  # 获取序列
        line = self.tokenizer.encode(line)  # 对序列进行编码
        return torch.tensor(line.ids)  # 返回编码后的序列张量


# 定义评估函数
def calculate_perplexity(logits, labels):
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    perplexity = torch.exp(loss)
    return perplexity.item()


def calculate_recovery_rate(predictions, labels):
    correct_predictions = (predictions == labels).sum().item()
    total_predictions = labels.numel()
    return correct_predictions / total_predictions


def median_recovery_score(recovery_scores):
    return np.median(recovery_scores)


def evaluate_model(model, dataloader, tokenizer):
    model.eval()
    total_loss = 0
    total_recovery_rate = 0
    recovery_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Evaluating"):
        # 将输入数据和标签移动到设备上（这是修改的部分）
        inputs = batch.to(device)
        labels = inputs.clone()

        with torch.no_grad():
            # 模型计算也在同一设备上进行（确保所有计算都在同一设备上）
            outputs = model(inputs, labels=labels)
            logits = outputs.logits
            loss = outputs.loss

        # 计算 perplexity 和 recovery_rate

        predictions = torch.argmax(logits, dim=-1)
        recovery_rate = calculate_recovery_rate(predictions, labels)

        recovery_scores.append(recovery_rate)

        total_loss += loss.item()
        total_recovery_rate += recovery_rate

    avg_loss = total_loss / len(dataloader)

    avg_recovery_rate = total_recovery_rate / len(dataloader)
    median_recovery = median_recovery_score(recovery_scores)



    return  avg_recovery_rate, median_recovery


def load_model_and_tokenizer(model_path, tokenizer_path):
    model = ProGenForCausalLM.from_pretrained(model_path)
    model.eval()

    tokenizer = Tokenizer.from_file(tokenizer_path)

    return model, tokenizer


def prepare_data(tokenizer, data_file, batch_size=9):#看你设备的性能调节batch_size
    with open(data_file, 'r') as f:
        lines = f.readlines()

    dataset = Protein_dataset(lines, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def main():
    model_path = "pretrainmodel"  # 替换为你的模型路径
    tokenizer_path = "E:\GIthubProject\ProGen2-finetuning-main\src\pretrainmodel/tokenizer.json"  # 替换为你的Tokenizer路径
    data_file = "test_data.txt"  # 替换为测试数据的文件路径

    # 加载模型和Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    print('loaded model and tokenizer successfully')
    # 准备数据
    dataloader = prepare_data(tokenizer, data_file)
    print('prepared data successfully')
    # 评估模型
    avg_recovery_rate, median_recovery = evaluate_model(model, dataloader, tokenizer)

    print(f"Average Recovery Rate: {avg_recovery_rate}")
    print(f"Median Recovery Score: {median_recovery}")


if __name__ == "__main__":
    main()
