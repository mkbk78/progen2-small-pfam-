import sys
import os
import argparse
import numpy as np
import re
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Protein_dataset(Dataset):
    def __init__(self, lines: list[str], tokenizer: Tokenizer):
        self.lines = lines  # 存储数据集中的序列
        self.tokenizer = tokenizer  # 用于对序列进行编码的 Tokenizer 对象

    def __len__(self):
        # 返回数据集中的序列数量
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]  # 获取序列
        line = self.tokenizer.encode(line)  # 对序列进行编码
        return torch.tensor(line.ids)  # 返回编码后的序列张量

def load_data(file: str) -> tuple[list[str], list[str]]: #加载数据并返回序列列表和前缀列表
    lines = []
    prefixes = set()
    # 打开文件并读取每一行
    with open(file, "r") as f:
        for line in f:
            # 去除每行两端的空格
            line = line.strip()
            # 使用正则表达式匹配前缀
            prefix = re.match(r"<\|.*\|>", line).group(0)
            # 将前缀添加到前缀集合中
            prefixes.add(prefix)
            # 将序列添加到序列列表中
            lines.append(line)
    # 将前缀集合转换为列表并排序
    prefixes = sorted(list(prefixes))
    # 返回序列列表和前缀列表
    return lines, prefixes

def init_new_embeddings(model: ProGenForCausalLM, prefixes: list[str]):#初始化新的嵌入向量
    if len(prefixes) <= 2:
        logger.info("没有新的嵌入向量需要初始化。")
        return
    # 创建一个新的嵌入矩阵，大小为 len(prefixes) - 2 x model.config.embed_dim
    new_embs = torch.zeros((len(prefixes) - 2, model.config.embed_dim)).to(model.device)

    # 获取未训练的嵌入向量（即权重矩阵的最后一个向量）
    unk_token_emb: torch.Tensor = model.transformer.wte.weight[-1].detach()
    # 计算未训练的嵌入向量的平均值和标准差
    mean_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.mean()
    std_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.std()

    # 使用正态分布初始化新的嵌入向量，与未训练的嵌入向量具有相同的分布
    torch.normal(mean_unk_emb, std_unk_emb, out=new_embs)
    # 将新的嵌入向量添加到已有的嵌入矩阵中
    new_embs = torch.cat([model.transformer.wte.weight, new_embs], dim=0)
    logger.debug(f"新的嵌入向量形状: {new_embs.shape}")
    # 更新模型的嵌入矩阵
    model.transformer.wte.weight = torch.nn.Parameter(new_embs, requires_grad=True)
    # 更新模型的词汇表大小
    model.config.vocab_size_emb = new_embs.shape[0]


def get_lr_schedule(optimizer: torch.optim.Optimizer, args: argparse.Namespace, train_steps: int):#获取学习率调度器

    if args.decay == "cosine":
        scheduler = get_cosine_schedule_with_warmup( # 使用余弦调度器
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=train_steps,
        )
    elif args.decay == "linear":
        scheduler = get_linear_schedule_with_warmup(# 使用线性调度器
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=train_steps,
        )
    elif args.decay == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(# 使用指数调度器
            optimizer, gamma=0.9, last_epoch=-1
        )
    elif args.decay == "constant":
        scheduler = get_constant_schedule_with_warmup(   # 使用常数调度器
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )
    else:
        raise ValueError(# 如果调度器类型无效，则抛出 ValueError
            f"Invalid learning rate decay type. Must be 'cosine', 'linear', 'exponential', or 'constant'. Got: {args.decay}"
        )
    return scheduler # 返回调度器


def train_epoch(
    model: ProGenForCausalLM,
    dataset: Protein_dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    args: argparse.Namespace,
):
    model.train()# 将模型设置为训练模式
    # 创建一个数据加载器，从 dataset 中获取数据，并设置批量大小为 args.batch_size，打乱数据
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0# 初始化损失为0
    pbar = tqdm(total=len(dataloader) // args.accumulation_steps)# 使用 tqdm 库创建一个进度条，显示总共有多少个 batch 需要训练
    batch: torch.Tensor# 遍历数据加载器中的每个 batch
    for i, batch in enumerate(dataloader):
        batch = batch.to(model.device)# 将 batch 数据移动到模型设备上（如 GPU）
        loss: torch.Tensor = model(batch, labels=batch).loss# 计算模型在当前 batch 上的损失
        loss = loss / args.accumulation_steps# 将损失除以 args.accumulation_steps 以进行梯度累积
        loss.backward()# 计算损失的梯度并反向传播
        total_loss = total_loss + loss.item()# 将损失累加到 total_loss 中
        if (i + 1) % args.accumulation_steps == 0:# 使用梯度累积来节省内存
            optimizer.step()# 更新模型参数
            scheduler.step()# 更新学习率
            optimizer.zero_grad()# 清空梯度
            pbar.update()# 更新进度条
    pbar.close()# 关闭进度条
    logger.info(f"TRAIN epoch {epoch}: loss: {total_loss / len(dataloader)}") # 记录日志，输出当前 epoch 的损失
    logger.debug(f"Last learning rate: {scheduler.get_last_lr()}")# 记录日志，输出最后一个学习率
    return total_loss / len(dataloader)# 返回当前 epoch 的平均损失


@torch.no_grad()
def evaluate(
    model: ProGenForCausalLM,
    dataset: Protein_dataset,
    args: argparse.Namespace,
    before_train: bool = False,
):#评估模型在给定数据集上的性能

    model.eval()# 将模型设置为评估模式
    total_loss = 0# 初始化损失为0

    if before_train:# 如果在训练之前进行评估
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)# batch_size 需要为 1，以便我们不会在张量中有不同长度的行
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size * 4, shuffle=True)# 否则，batch_size 为 args.batch_size * 4
    total_length = len(dataloader) # 获取数据加载器的长度
    pbar = tqdm(total=total_length)# 使用 tqdm 库创建一个进度条，显示总共有多少个 batch 需要评估
    batch: torch.Tensor  # 遍历数据加载器中的每个 batch
    for batch in dataloader:
        if before_train:# 如果在训练之前进行评估
            non_zero_length = (batch != 0).sum().item()# 移除填充，因为基础模型没有用填充进行训练
            batch = batch[:, :non_zero_length]
        batch = batch.to(model.device)# 将 batch 数据移动到模型设备上（如 GPU）
        loss: torch.Tensor = model(batch, labels=batch).loss# 计算模型在当前 batch 上的损失
        total_loss += loss.item()# 将损失累加到 total_loss 中
        pbar.update()# 更新进度条
    pbar.close()# 关闭进度条
    logger.info(f"EVAL loss: {total_loss / total_length}")  # 记录日志，输出评估的损失

    return total_loss / total_length    # 返回评估的平均损失


def train(
        model: ProGenForCausalLM,
        tokenizer: Tokenizer,
        train_dataset: Protein_dataset,
        test_dataset: Protein_dataset,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        args: argparse.Namespace,
        job_id: str,
):
    # 初始化训练损失和评估损失列表
    train_losses = []
    eval_losses = []
    for epoch in range(1, args.epochs + 1):  # 遍历每个 epoch
        logger.info(f"Start time of epoch {epoch}: {datetime.now()}")# 记录当前 epoch 的开始时间
        # 训练当前 epoch
        train_loss = train_epoch(model, train_dataset, optimizer, scheduler, epoch, args)
        train_losses.append(train_loss)       # 将训练损失添加到列表中

        logger.info(f"Running test set evaluation after {epoch} epochs:")      # 记录评估开始
        eval_loss = evaluate(model, test_dataset, args)        # 评估当前 epoch
        eval_losses.append(eval_loss)        # 将评估损失添加到列表中

        model_name = (job_id + "-" if job_id is not None else "") + args.model.strip(os.pathsep).split(os.pathsep)[-1]# 获取模型名称
        if epoch % args.checkpoint_rate == 0 or epoch == args.epochs:# 如果当前 epoch 是 checkpoint_rate 的倍数或者当前 epoch 是最后一个 epoch
            # 创建 checkpoint 路径
            checkpoint_path = os.path.join("checkpoints", f"{model_name}-finetuned", f"e{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)

            # 保存模型和分词器
            model.save_pretrained(checkpoint_path)
            tokenizer.save(os.path.join(checkpoint_path, "tokenizer.json"), pretty=True)

            if args.save_optimizer: # 如果需要保存优化器和调度器
                logger.info("Saving optimizer and scheduler...")
                # 保存优化器和调度器的状态
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

            logger.info(f"Model saved at: {checkpoint_path}")# 记录模型保存路径
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')# 保存模型状态
    return model, train_losses, eval_losses# 返回模型、训练损失和评估损失


def main(args: argparse.Namespace):

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 获取 Slurm 作业 ID
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        logger.debug(f"Slurm job id: {job_id}")
    else:
        logger.warning("No Slurm job ID found.")

    # 加载数据和分词器
    tokenizer: Tokenizer = Tokenizer.from_pretrained(args.model)
    tokenizer.enable_padding(
        direction="right", pad_id=0, pad_token="<|pad|>", length=1024
    )
    tokenizer.enable_truncation(max_length=1024)

    # 加载训练数据和测试数据
    train_data, prefixes = load_data(args.train_file)
    test_data, prefixes_test = load_data(args.test_file)
    logger.info(f"Found prefixes: {prefixes}")
    assert prefixes == prefixes_test, "Prefixes in train and test data must be the same"
    tokenizer.add_tokens(prefixes)

    # 创建训练数据和测试数据集
    train_data = Protein_dataset(train_data, tokenizer)
    test_data = Protein_dataset(test_data, tokenizer)
    logger.debug(f"Train data size: {len(train_data)}")
    logger.debug(f"Test data size: {len(test_data)}")

    # 检查 CUDA 是否可用
    print(torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU. Please consider using a GPU for training.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # 打印超参数
    logger.debug(f"hyperparameters: effective batch={args.batch_size * args.accumulation_steps}, {args.batch_size=}, {args.accumulation_steps=}, {args.epochs=}, {args.lr=}, {args.warmup_steps=}, {args.checkpoint_rate=}")

    # 加载模型
    logger.info(f"Loading model: {args.model}...")
    model = ProGenForCausalLM.from_pretrained(args.model).to(device)
    logger.info(f"Model loaded. Parameter count: {model.num_parameters() // 1e6} M")
    init_new_embeddings(model, prefixes)

    # 创建优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_steps = (
        args.epochs * len(train_data) // (args.batch_size * args.accumulation_steps)
    )
    if training_steps > 0:
        logger.debug(f"Weight updates per epoch: {training_steps / args.epochs}")
    logger.debug(f"Total weight updates: {training_steps}")
    scheduler = get_lr_schedule(optimizer, args, training_steps)

    # 在训练之前进行评估
    if args.eval_before_train:
        logger.info("Runnning evaluation on test set before training...")
        evaluate(model, test_data, args, before_train=True)

    # 训练模型
    model, train_losses, test_losses = train(
        model,
        tokenizer,
        train_data,
        test_data,
        optimizer,
        scheduler,
        args,
        job_id,
    )

    logger.info("Finetuning finished.")
    logger.info(f"Train losses: {train_losses}")
    logger.info(f"Test losses: {test_losses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hugohrban/progen2-small",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
   )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=200,
    )
    parser.add_argument("--checkpoint_rate", type=int, default=5,)
    parser.add_argument(
        "--decay",
        type=str,
        choices=["cosine", "linear", "constant"],
        default="cosine",
    )
    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--eval_before_train",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    main(args)
