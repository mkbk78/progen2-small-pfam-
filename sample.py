import os
import argparse
import logging

import torch
from tqdm import tqdm
import re
from typing import Optional, Union

from tokenizers import Tokenizer, Encoding
from models.progen.modeling_progen import ProGenForCausalLM, ProGenPreTrainedModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(
    model: ProGenForCausalLM,
    tokenizer: Tokenizer,
    device: torch.device,
    prompt: Union[str, torch.Tensor],
    max_length: int,
    num_return_sequences: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> list[str]:
    #从模型中生成样本，使用 top-k 采样和温度参数。
    model.eval()# 将模型设置为评估模式
    if isinstance(prompt, str):# 如果 prompt 是字符串
        encoding: Encoding = tokenizer.encode(prompt)# 使用分词器将 prompt 编码为 token ids
        ids = torch.tensor(encoding.ids)
        ids = ids[:len(torch.nonzero(ids))]# 移除 padding

        x = torch.zeros((num_return_sequences, ids.shape[0]))# 创建一个形状为 (num_return_sequences, ids.shape[0]) 的张量，用于存储生成的样本
        x = x + ids# 将 prompt 的 token ids 复制到生成的样本中
        x = x.to(device).to(torch.int32)# 将生成的样本移动到设备上
    
    elif isinstance(prompt, torch.Tensor):# 如果 prompt 是一个张量
      
        x = prompt.to(device).to(torch.int32)     # 将生成的样本移动到设备上 
    else:# 如果 prompt 既不是字符串也不是张量，则抛出 ValueError
        raise ValueError("Prompt should be either string or torch.Tensor")

    past_key_values = None  # 初始化 past_key_values
    generated = x   # 初始化生成的样本

    pbar = tqdm(total=max_length - generated.shape[-1])     # 创建一个进度条，显示总共有多少个 token 需要生成
    while generated.shape[-1] < max_length:                 # 当生成的样本长度小于最大长度时，继续生成
        output = model(x, past_key_values=past_key_values)  # 使用缓存的注意力输出
        past_key_values = output.past_key_values            # 更新 past_key_values
        logits = output.logits                              # 获取 logits                                 
        logits = logits[:, -1, :]                           # 只获取最后一个 token 的 logits                        
        logits = logits / temperature                       # 将 logits 除以温度
        if top_k is not None:                               # 如果 top_k 不为 None，则使用 top-k 采样
            v, _ = torch.topk(logits, top_k, dim=-1)             # 获取 top-k 的 logits                
            logits = torch.where(logits >= v[:, -1:], logits, -1e9)      # 将 logits 中小于 top-k 的 logits 设置为 -1e9
        probs = torch.softmax(logits, dim=-1)          # 计算概率                
        x = torch.multinomial(probs, num_samples=1)                 # 使用 multinomial 采样           
        generated = torch.cat([generated, x], dim=-1)             # 将生成的样本添加到生成的样本中     
        pbar.update()        # 更新进度条
    pbar.close()    # 关闭进度条
    decoded = [tokenizer.decode(row.detach().cpu().numpy().tolist()) for row in generated]    # 使用分词器将生成的样本解码为字符串
    return decoded    # 返回解码后的字符串


def truncate(seq: str) -> str:
    #移除家族特殊标记，初始的 1 或 2 个标记，并截断序列到第一个找到的 1 或 2 个标记。以 2 开头的序列（C -> N 生成）会被反转。
    
    seq = re.sub(r"<\|.*\|>", "", seq)# 移除家族标记
    terminus = seq[0]# 移除初始终止符
    seq = seq[1:]

    min_1 = seq.find("1") # 找到第一个 "1" 的位置
    if min_1 == -1:# 如果找不到 "1"，则将 min_1 设置为序列的长度
        min_1 = len(seq)

    min_2 = seq.find("2")# 找到第一个 "2" 的位置
    if min_2 == -1:# 如果找不到 "2"，则将 min_2 设置为序列的长度
        min_2 = len(seq)

    seq = seq[: min(min_1, min_2)]    # 截断序列到第一个找到的 "1" 或 "2" 的位置
    if terminus == "1":    # 如果初始终止符是 "1"，则返回截断后的序列
        return seq
    else:    # 否则，反转截断后的序列并返回
        return seq[::-1]


def reverse(seq: str) -> str:    
    #反转一个以家庭标记和初始终止符开头的序列。然后，继续在相反的方向生成序列。
    prefix_pattern = re.compile(r"<\|.*\|>")    # 创建一个正则表达式，用于匹配家庭标记
    m = re.search(prefix_pattern, seq)    # 在序列中搜索家庭标记
    prefix = m.group() if m else ""    # 如果找到家庭标记，则将其保存到 prefix 变量中
    seq = seq.replace(prefix, "")  # 移除家庭标记
    
    # 移除初始终止符并反转序列
    start_terminus = seq[0]
    seq = seq[1:]
    seq = seq[::-1]

    # 如果我们生成了序列的末尾
    if seq[0] in ["1", "2"]:
        seq = seq[1:]

    # 反转并添加相反的终止符
    if start_terminus == "1":
        return prefix + "2" + seq
    else:
        return prefix + "1" + seq


def main(args):
    # 设置环境变量
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 检查 CUDA 是否可用
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # 如果使用 CPU 并且 batch_size 较大，则发出警告
    if str(device) == "cpu" and args.batch_size > 1:
        logger.warning(f"You are using CPU for inference with a relatively high batch size of {args.batch_size}, therefore inference might be slow. Consider using a GPU or smaller batch.")

    # 加载模型
    model = ProGenForCausalLM.from_pretrained("pretrainmodel",ignore_mismatched_sizes=True ).to(device)
    logger.debug("Model loaded.")

    # 加载分词器
    logger.info("Loading tokenizer")
    tokenizer: Tokenizer = Tokenizer.from_pretrained(args.model)
    tokenizer.no_padding()
    logger.debug("Tokenizer loaded.")
    logger.debug(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    logger.debug(f"Tokenizer vocab: {tokenizer.get_vocab()}")

    # 创建样本目录
    samples_dir = os.path.join("generated_samples")
    os.makedirs(samples_dir, exist_ok=True)
    # 创建输出文件
    output_file = os.path.join(samples_dir, f"samples_ctx{args.prompt}_k{args.k}_t{args.t}.fa")
    print(output_file)

    # 如果 top_k 超过词汇表大小，则设置为 None
    if args.k == 0 or args.k > model.config.vocab_size_lm_head:
        args.k = None

    # 打印采样参数
    logger.debug(f"Sampling parameters: top_k={args.k}, temperature={args.t}")
    # 编码 prompt
    tokens = tokenizer.encode(args.prompt).tokens
    logger.info(f"Prompt tokens: {tokens}")

    # 如果使用双向采样，则设置最大长度
    if args.bidirectional:
        args.max_length = (args.max_length - len(tokens)) // 2
        if len(tokens) <= 2:
            logger.warning("Prompt is too short for bidirectional sampling. Please provide a longer prompt.")


    with open(output_file, "w") as f:    # 打开输出文件
        for i in range(args.iters):       # 遍历每个迭代
            logger.info(f"Sampling batch {i+1} / {args.iters}")
            samples = sample(      # 生成样本
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=args.prompt,
                num_return_sequences=args.batch_size,
                temperature=args.t,
                max_length=args.max_length,
                top_k=args.k,
            )
            if args.bidirectional:            # 如果使用双向采样
                reversed_samples = [reverse(s) for s in samples]         # 反转样本
                samples = []
                for rs in reversed_samples:
                    prompt = torch.tensor(tokenizer.encode(rs).ids).view(1, -1).to(model.device)      # 编码反转后的样本
                    samples.extend(sample(# 生成样本
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        num_return_sequences=1,
                        temperature=args.t,
                        max_length=args.max_length * 2,
                        top_k=args.k,
                    ))
            # 将样本写入文件
            for j, c in enumerate(samples):
                print(f">seq_{i * args.batch_size + j+1}", file=f)
                c = truncate(c)
                print(c, file=f)
    logger.info(f"Generated samples were saved to file {output_file}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="hugohrban/progen2-small",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--prompt",
        type=str,
        default="1",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
    )
    parser.add_argument("--t", type=float, default=1.0, )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true",)
    parser.add_argument(
        "--bidirectional",
        action="store_true", 
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)
    logger.info("Sampling finished.")
