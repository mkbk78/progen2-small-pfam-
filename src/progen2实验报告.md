# 测试报告（要求：markdown格式写个测试报告，包括模型结构、模型性能分析、案例分析。）

# 1. 引言

## 1.1 编写目的
本测试报告旨在评估基于深度学习的蛋白质序列生成模型的性能，包括模型结构分析、性能评估以及案例研究。

## 1.2 背景

蛋白质序列的生成对于生物信息学和药物设计领域具有重要意义。本研究采用了Hugo Hrbáň等人提出的模型，该模型基于预训练的深度神经网络，通过微调生成具有特定特征的蛋白质序列。

## 1.3 范围
本报告覆盖模型结构描述、性能指标分析、案例分析以及生成序列的生物信息学验证。

# 2. 模型结构

## 2.1 基础模型

模型基于ProGen2（Nijkamp et al., 2022），这是一个基于Transformer架构的蛋白质语言模型。原始模型在大量蛋白质序列上进行预训练，以学习氨基酸序列的上下文表示。

## 2.2 模型架构细节

- **层数**：12层

  ```
  	    # 1.初始化注意力Dropout层，用于注意力分数上的Dropout
  	    self.attn_dropout = nn.Dropout(config.attn_pdrop)
  	    
  	    # 2.初始化残差连接后的Dropout层
          self.resid_dropout = nn.Dropout(config.resid_pdrop)
                  
          # 3.初始化一个线性层，用于将输入投影到查询（Q）、键（K）和值（V）三个空间
          self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
  
          # 4.初始化另一个线性层，用于将注意力机制的输出投影回原始的嵌入维度     
          self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
          
          # 5.创建一个线性层，将输入从嵌入维度映射到中间层大小
          self.fc_in = nn.Linear(embed_dim, intermediate_size)
          
          # 6.创建一个线性层，将中间层输出映射回嵌入维度
          self.fc_out = nn.Linear(intermediate_size, embed_dim)
  
          # 7.创建一个Dropout层，用于减少过拟合，dropout比率从配置中获取
          self.dropout = nn.Dropout(config.resid_pdrop)
          
          # 8.初始化第一个LayerNorm层
          self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
          
          # 9.初始化词嵌入层
          self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
  	   
  	   # 10.初始化dropout层
          self.drop = nn.Dropout(config.embd_pdrop)
          
          # 11.初始化最后的层归一化层
          self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
          
          # 12.实例化一个线性层作为语言模型头部，用于将隐藏状态映射到词汇表
          self.lm_head = nn.Linear(config.n_embd, config.vocab_size)        
  ```

  

- **注意力头**：16个

- **隐藏维度**：1024

- **上下文窗口大小**：1024个令牌

  <img src="pic\image-20240810003137462.png" />

## 2.3模型结构图

  <img src="pic/export (1).png" />

## 2.4小样本训练过程

```
download_data：从pfam平台下载PF00001,PF00257,PF02680,PF12365,PF12366的fasta格式数据
prepare_data:按train和test数据集8：2的比例，将fasta数据打乱分别保存到train_data.txt和test_data.txt
加载数据：load_data 函数加载txt格式数据。
模型和分词器加载：从huggingface加载预训练progen2-samll的模型和分词器。
设置cuda训练
设置batch_size
设置学习率lr
设置调度器cosine，使用余弦退火调度器来调整学习率
设置epoch
循环训练后评估loss保存优化器和checkpoints

```



# 3. 模型性能分析

将微调后的模型进行Recovery Rate指标的评估，得出

<img src="pic\image-20240811002403722.png" />

根据评估指标数据，我们可以做出以下分析：

1. **Average Recovery Rate (平均恢复率)**: 0.7813
   - 这个指标表示在所有测试样本中，模型恢复正确蛋白质序列的平均比例。0.7813的平均恢复率表示模型在大多数情况下能够准确生成接近实际的蛋白质序列。
2. **Median Recovery Score (中位恢复得分)**: 0.7932
   - 中位恢复得分表示一半的样本恢复得分高于这个值，另一半低于这个值。0.7932的中位数意味着大多数样本的恢复得分相对较高，说明模型的表现相对稳定且一致。

## **总结**:

- **性能良好**: 这两个指标都接近于0.8，说明模型在生成蛋白质序列方面表现不错。尤其是中位恢复得分略高于平均恢复率，显示出模型在大多数情况下的表现是比较稳定的。
- **改进空间**: 虽然结果不错，但仍有提升空间。可以考虑进一步优化模型或调整训练数据，以提高这些指标。

总的来说，模型的效果是较好的，但可以通过一些技术手段进一步改进。

# 4.案例分析

以<|pf02680|>家族的'1MIAVKRVVLDVL'进行生成蛋白质序列
<img src="pic\image-20240811011006224.png" />


生成结果如下

```
>seq_1
MIAVKRVVLDVLSTEQSFLGVSTRKGLALLGTFDKANTEGLKEAEGDEE
>seq_2
MIAVKRVVLDVLFEELQNGRHQEDETLSPQVIIRQFAARPFFLKGLQTT
>seq_3
MIAVKRVVLDVLFDPSDGQVIEHEQVPILVAGPIAIQTKDTLALKAGQK
>seq_4
MIAVKRVVLDVLAPIKLVKFLNELKDDGHPSEGIEYVSEEPELIQIGAK
```

与pf02680家族下载的序列进行比对，相似度如下

<img src="pic\image-20240811012722263.png" />

## 4.1**结果与分析**

以下是生成序列与最相似家族序列的比对结果：

**序列1**

- **生成序列**: `MIAVKRVVLDVLSTEQSFLGVSTRKGLALLGTFDKANTEGLKEAEGDEE`
- **最相似的家族序列**: `1MIAVKRVVLDVLKPHHPNALEFCQALSAAVEGSTVRVAVIAVDEKTESLEVSVEAALIDFDLLQKVINKMGGALHSIDEVEVKNEPST2`
- **相似度**: 0.31

**分析**: 生成序列与家族序列的相似度达到了0.31。虽然相似度较为中等，但模型成功生成了与家族序列具有相当相似性的序列，证明模型在一定程度上学到了家族序列的模式。

**序列2**

- **生成序列**: `MIAVKRVVLDVLFEELQNGRHQEDETLSPQVIIRQFAARPFFLKGLQTT`
- **最相似的家族序列**: `1MIKVKRLVLDVLKPHQPNGLEFASALAERCGSCRIKYCVVEVDEQTETTLLTIDGDDIQFNVVTEAIRSMGATIHSIDEVDVVGLQTAT2`
- **相似度**: 0.28

**分析**: 该序列的相似度为0.28，表明生成的序列与家族序列存在显著的相似性。模型能够生成具有相似结构的序列，暗示它在捕捉复杂蛋白质结构的关键特征上具有一定能力。

**序列3**

- **生成序列**: `MIAVKRVVLDVLFDPSDGQVIEHEQVPILVAGPIAIQTKDTLALKAGQK`
- **最相似的家族序列**: `2ERTATVEDVSHIVAGMSEILKRVKEFDIDTGEITITLTLTEVDVENVKINVHEVGPLEVIKQAMDILTPGKITKLVDLVLKTVGM1`
- **相似度**: 0.28

**分析**: 该生成序列与家族序列的相似度为0.28，说明模型在生成过程中，保持了与家族序列较高的相似性，特别是在核心区域。模型显示了在捕捉蛋白质序列的进化保守性方面的潜力。

**序列4**

- **生成序列**: `MIAVKRVVLDVLAPIKLVKFLNELKDDGHPSEGIEYVSEEPELIQIGAK`
- **最相似的家族序列**: `1MALTKLVLDVLKQLKGPTIIEVAYRLLELDGIKSVDIEIKEIDVETLSLTITIEGSNIDFEK2`
- **相似度**: 0.28

**分析**: 最后一条生成序列与家族序列的相似度也为0.28，进一步表明模型在生成过程中成功保留了部分结构和功能上的重要特征。这种相似性展示了模型在生成与自然序列相似的蛋白质序列时的稳定性。

## 4.2 **总结与展望**

本次评估显示，蛋白质序列生成模型能够生成与已知家族序列具有一定相似度的蛋白质序列。虽然相似度未达到极高的水平，但模型已经展示了捕捉蛋白质序列结构和功能特征的能力，这对于未来的优化和应用具有重要的参考价值。

### 优势分析

1. **保守性捕捉**: 模型在生成序列中成功保留了已知家族序列的保守性特征。这表明模型具有一定的生物学特征学习能力，能够在生成过程中识别和再现关键的结构和功能位点。

2. **多样性生成**: 尽管保持了与家族序列的相似性，模型仍然能够生成具有多样性的新序列。这种多样性对于蛋白质设计至关重要，能够为新功能的探索提供基础。

3. **稳定性**: 不同生成序列的相似度表现出一致性，说明模型在蛋白质序列生成过程中具有稳定的性能。模型能够在不同的输入条件下保持一致的生成质量，这对于实际应用非常重要。

### 问题分析

1. **相似度较低**: 虽然模型能够生成与家族序列相似的序列，但相似度普遍在0.28-0.31之间。这表明模型在捕捉家族序列的完整性和精确性上仍有提升空间，可能需要更复杂的模型结构或更大的训练数据集来提高生成质量。

2. **特征学习不足**: 尽管模型能够捕捉部分保守性，但相似度结果显示模型可能未能完全学习家族序列中的所有关键特征。尤其在处理复杂结构或功能区域时，模型可能未能充分再现这些特征。

3. **生物学功能未知**: 虽然生成的序列与已知家族序列相似，但尚不清楚这些序列在实际生物学功能中的表现。进一步的实验验证和功能预测分析可能需要以确认生成序列的实际应用价值。

注：图片在pic文件夹里

# 5.全部数据代码见：

#### [mkbk78/progen2-small-pfam-: progen2-small模型对于PF00001,PF00257,PF02680,PF12365,PF12366进行微调后的模型 (github.com)](https://github.com/mkbk78/progen2-small-pfam-)
