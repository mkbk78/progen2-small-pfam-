import os
import argparse
import logging
import re
from typing import Optional, Union
from Bio import SeqIO
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(input_file_name: str, bidirectional: bool = False) -> list[str]:
    m = re.search(r"PF([0-9]{5})\.fasta", input_file_name) # 使用正则表达式从输入文件名中查找 PFXXXXX.fasta 格式的字符串，其中 X 是数字
    if m is None:# 如果找不到这样的字符串，则记录错误日志，并抛出一个 ValueError 异常
        logger.error("Invalid input file name.")
        raise ValueError("Invalid input file name. Input file should be in the format PFXXXXX.fasta. Please consider downloading the data using the download_pfam.py script.")

    prefix = m.group(1)# 从 m 中获取第一个捕获组，即文件名中的 PF 后面的数字，并将其赋值给 prefix 变量
    seqs = SeqIO.parse(open(input_file_name, "r"), "fasta")# 使用 SeqIO.parse 函数从打开的输入文件中解析 fasta 格式的序列，并将其赋值给 seqs 变量
    parsed_seqs = []# 创建一个空列表 parsed_seqs，用于存储处理后的序列
    for s in seqs:# 遍历 seqs 中的每个序列
        parsed_seqs.append(f"<|pf{prefix}|>1{str(s.seq)}2")# 将一个处理后的序列添加到 parsed_seqs 列表中
        if bidirectional:# 如果 bidirectional 参数的值为 True，则将一个反向互补的序列添加到 parsed_seqs 列表中
            parsed_seqs.append(f"<|pf{prefix}|>2{str(s.seq)[::-1]}1")

    return parsed_seqs # 返回 parsed_seqs 列表，其中包含了处理后的序列


def main(args: argparse.Namespace):
    np.random.seed(args.seed)# 设置随机数种子，以便结果可重复

    if not 0 <= args.train_split_ratio <= 1: #检查 train_split_ratio 的值是否在 0 到 1 之间
        raise ValueError("Train-test split ratio must be between 0 and 1.")

    # 创建两个空列表 train_data 和 test_data，分别用于存储训练数据和测试数据
    train_data = []
    test_data = []

    for input_file in args.input_files:# 遍历命令行参数 args.input_files 中的每个输入文件
        data = prepare_data(input_file, args.bidirectional)# 调用 prepare_data 函数处理输入文件，并将返回的数据赋值给 data 变量
        logging.info(f"Loaded {len(data)} sequences from {input_file}")# 记录日志，输出加载的序列数量
        np.random.shuffle(data) # 打乱数据
        split_idx = int(len(data) * args.train_split_ratio)# 计算分割索引
        # 将数据分割成训练集和测试集
        train_data.extend(data[:split_idx])
        test_data.extend(data[split_idx:])

    np.random.shuffle(train_data)# 打乱训练集
    np.random.shuffle(test_data)# 打乱测试集


    if args.bidirectional:# 如果 bidirectional 参数的值为 True，则记录日志，提示数据是双向的
        logging.info("Data is bidirectional. Each sequence will be stored in both directions.")

    # 记录日志，输出训练数据和测试数据的长度
    logging.info(f"Train data: {len(train_data)} sequences")
    logging.info(f"Test data: {len(test_data)} sequences")

    # 将训练数据保存到文件
    logging.info(f"Saving training data to {args.output_file_train}")
    with open(args.output_file_train, "w") as f:
        for line in train_data:
            f.write(line + "\n")

    # 将测试数据保存到文件
    logging.info(f"Saving test data to {args.output_file_test}")
    with open(args.output_file_test, "w") as f:
        for line in test_data:
            f.write(line + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files", type=str, nargs="+", required=True, help="输入fasta格式数据文件."
    )
    parser.add_argument(
        "--output_file_train", type=str, default="train_data.txt", help="输出训练数据文件名"
    )
    parser.add_argument(
        "--output_file_test", type=str, default="test_data.txt", help="输出训练测试文件名"
    )
    parser.add_argument(
        "--bidirectional",
        "-b",
        action="store_true",
        help="是否存储序列的逆序. Default: False.",
    )
    parser.add_argument(
        "--train_split_ratio",
        "-s",
        type=float,
        default=0.8,
        help="训练和测试的比例. Default: 0.8",
    )
    parser.add_argument(
        "--seed", type=int, default=69, help="随机种子",
    )
    args = parser.parse_args()
    main(args)