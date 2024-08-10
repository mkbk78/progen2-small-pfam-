```bash
cd src
python3 -m venv venv
pip3 install -r requirements.txt
```

### 下载数据（pfam）

```bash
python3 download_pfam.py PF00001 PF00257 PF02680 PF12365 PF12366
```

### 转换格式

```bash
python3 prepare_data.py --input_files downloads/PF00001.fasta downloads/PF00257.fasta  downloads/PF02680.fasta downloads/PF12365.fasta downloads/PF12366.fasta --output_file_train=train_data.txt --output_file_test=test_data.txt --train_split_ratio=0.8 --bidirectional
```

### 进行微调

```bash
python3 finetune.py --model=hugohrban/progen2-small --train_file=train_data.txt --test_file=test_data.txt --device=cuda --epochs=4 --batch_size=9 --accumulation_steps=4 --lr=1e-4 --decay=cosine --warmup_steps=200 --eval_before_train
```

### 生成蛋白质序列

```bash
python3 sample.py --model=hugohrban/progen2-small --device=cuda --batch_size=8 --iters=1 --max_length=512 --t=1.0 --k=10 --prompt="1MEVVIVTGMSGAGK"
```

### 模型评估
```bash
python3 evaluate.py
```
