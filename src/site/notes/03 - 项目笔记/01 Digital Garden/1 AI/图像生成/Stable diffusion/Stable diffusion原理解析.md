---
{"dg-publish":true,"permalink":"/03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Stable diffusion原理解析/","noteIcon":""}
---


笔记由 B 站大佬 ZHO 的[科普视频](https://www.bilibili.com/video/BV1BC4y1V7u9) 学习整理而成。

# LLM 进化树

![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Fuw9fv9akAA_h0q(1) 3.jpg](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Fuw9fv9akAA_h0q(1)%203.jpg)

# 图像生成模型

- GAN 生成对抗模型， 生成模型 VS 判别模型
- VAE
- Flow
- Diffusion Model

#todo/图像生成论文: 《what are diffusion models》论文详细阅读，介绍了更多的图像生成模型

# Diffusion models

## 两个过程

- 正向扩散：通过添加噪声来扰动数据分布
- 反向传播：将噪声从数据中去除，生成图像
  #todo/图像生成论文: 《Diffusion Models: A Comprehensive Survey of Methods and Applications》

## 扩散模型的三大基础

- DDPM
- SGM
- Scope SDE

## 扩散模型的分类

- 高效采样
- 最大似然增强
- 结构化数据

## 扩散模型的六大应用领域

- 计算机视觉
- 自然语言生成
- 时态数据建模
- 多模态学习
- 鲁棒学习
- 跨学科学习

#todo/视频课程: Google deenlearningAI, 《How Diffusion Models Work》

# Stable Diffusion 发展线梳理

![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250707155746.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250707155746.png)

## 主线

### 基础

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585)：
- [DDPM](https://arxiv.org/pdf/1907.05600v1.pdf)：15 年提出的扩散概率模型，经过 ddpm 的改进之后，逐渐进入人们的视野。ddpm 将“去噪”扩散概率模型应用到图像中，并且通过固定前向过程的方差、选择合适的反向过程参数化，以及使用高斯分布来表示条件概率，设计出了简单高效的扩散模型
- [SDM](https://arxiv.org/pdf/2006.11239v1.pdf)
- [Scope SDE](https://arxiv.org/pdf/2011.13456v1)
- DDPM、SDM、Scope SDE 三大基础论文，奠定了扩散概率模型在图像生成领域的基础
  - 基于分数的生成模型 SGM 和扩散概率模型 DDPM 都可以视为由分数函数确定的随机微分方式 Score SDE 的离散化
  - 分数生成模型和扩散概率模型被连接到一个统一的框架中

### 高效采样

- 2020 年 10 月，DDIM 被提出，弃用马尔科夫链，采样速度提升 50 倍，反向降噪转变为确定性过程，首次拥有一致性
- 2021 年 2 月，OpenAI 提出 improved-diffusion，生成速度和质量有了比较大的飞跃。

### 引导扩散

- 2021 年 5 月，OpenAI 发布[《diffusionmodel beats GANs on image synthesis》](https://arxiv.org/pdf/2105.05233)，提出 guided-diffusion，通过增加分类器作为引导，用多样性换取写实性。但是引导函数与扩散模型是分开训练的，无法联合训练。
- 2021 年底，DDPM 作者提出了无需额外的分类器的扩散引导方法，[classifier-free diffusion guidance](https://arxiv.org/pdf/2207.12598), CFG(即后来 SD 中的 CFG 参数)。训练模型的成本大大增加
- 2021 年 12 月，OpenAI 的新论文[GLIDE](https://arxiv.org/pdf/2112.10741v1.pdf)，对比了用 CLIP 模型做引导，和 CFG 无分类器做引导的方法，后者更受青睐，虽然训练成本更高。 OpenAi 训练出基于无分类器引导的全新模型 GLIDE，拥有 35 亿参数量的超大规模文生图扩散模型，也可用于图像修复，实现了基于文本驱动的图像编辑功能

### Clip + Diffusion

- 2022 年 4 月，[DALL·E 2](https://arxiv.org/pdf/2204.06125)诞生，基本思想是 CLIP 和 GLIDE 的结合，模型名取自萨尔瓦多达利（Salvador Dali)和 Wall·E. unClip，反转 Clip 编码器
  - 第一阶段的先验模型接受编码后的文本嵌入作为输入，经过 prior 采样之后生成一个 CLIP 图像嵌入
  - 第二阶段的解码器，以上一阶段的图像嵌入作为输入来生成最终图像
  - 实现了输入既可以是文字，也可以是图像。
  - 此时扩散模型可以从现有图像中获取概念，并基于此概念而生存保留原本语义和风格的图像变体
  - ==图生图与 unCLIP 的区别==：
    - 图生图（图像作为噪声进入模型）：构图等方面与原图相似，仅在细节方面产生变化，适合图像做**微调**、**修复**等场景
    - unCLIP（图像作为提示词进入模型）：输入的图被编码后变为提示词输入到模型中，对于模型来说，是**提取概念**后重新生成的过程，构图与细节与原图不同
    - ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250707185800.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250707185800.png)
    - ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250707191533.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250707191533.png)
      ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250707185403.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250707185403.png)
  - ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250707191913.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250707191913.png)
  - ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250707191929.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250707191929.png)
    ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250707184808.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250707184808.png)
- 同年 5 月，Google 推出语言模型 T5 和 DM 结合的新模型[Imagen](https://arxiv.org/pdf/2205.11487)

### 潜空间扩散模型 LDM

- 21 年 12 月，LDM V1
- 22 年 4 月， LDM V2，14.5 亿参数量的 LDM 模型
  - 将原本在像素层面运行的扩散模型降低维度到[[03 - 项目笔记/01 Digital Garden/1 AI/机器学习与神经网络/潜空间\|潜空间]]进行，降低训练和运算成本
  - 增加了交叉注意力层，支持多种条件输入

### SD

- 2022 年 8 月，[SD](https://github.com/COMPVIS/STABLE-DIFFUSION)公开版正式发布，CompVis 牵头，stability AI、runaway 等共同合作，以 LDM 架构为基础，在 LAION-5B 的子集上完成训练，可以生成 512 \* 512 分辨率的图像。使用 CLIP VIT-L 作为文本编码器。可以在 10GB 级别显卡上运行
- 2022 年 11 月底，stability AI 宣布推出[SD 2.0](https://github.com/stability-ai/stablediffusion)
- 12 月初推出 SD 2.1，图像模型进入以 SD 为核心的 LDM 时代

### 微调

- 22 年 8 月，英伟达提出的 embedding 模型，[An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/pdf/2208.01618v1)
- 同月，谷歌发布论文[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/pdf/2208.12242v1)，进入个性化微调时代
- 2022 年 12 月， [LoRA for DM](https://github.com/cloneofsimo/lora)用于扩散模型的微调

### 控制

- [GLIGEN](https://arxiv.org/pdf/2301.07093v1)，擅长指定位置生成
  - 文本+框，图像+框，图像风格+文本+框，文本实体+关键点
    ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250708181439.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250708181439.png)
    ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250708182246.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250708182246.png)
- Control Net
- T2I-Adapter：对生成速度有要求或者算力有限，建议使用 T2I

### SDXL 新架构

- 23 年 4 月份，stability ai 推出 SDXL beta, 6 月推出 SDXL 0.9 [SDXL](https://arxiv.org/pdf/2307.01952)，新架构模块化，新增精调模型
- 7 月底正式发布 SDXL 1.0, 可以生成 1024 \* 1024 高清图

### 后续发展

- stability ai 推出谷歌 Imagen 的复现模型[IF](https://github.com/deep-floyd/IF)，可以解决**文字生成、空间位置关系理解**等问题
- 2023 年 5 月 stability ai 推出文生动画模型 stable animation SDK

### image prompt （类 unclip）

- [Revision](https://huggingface.co/stabilityai/control-lora):Revision is a novel approach of using images to prompt SDXL.
- 腾讯[Ip-adapter](https://arxiv.org/pdf/2308.06721)
  ![03 - 项目笔记/01 Digital Garden/1 AI/图像生成/Stable diffusion/Pasted image 20250708191854.png](/img/user/03%20-%20%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0/01%20Digital%20Garden/1%20AI/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90/Stable%20diffusion/Pasted%20image%2020250708191854.png)

## 辅线

### CLIP

- 2021 年 2 月，OpenAI 提出拥有超强零样本迁移能力的 CLIP 模型，图像与文本之间的向量转化

### 微调

- 2016 年 9 月，语言模型微调的[hypernetworks](https://arxiv.org/pdf/1609.09106v1)
- 2021 年 6 月，微软提出[LoRA](https://arxiv.org/pdf/2106.09685v1)用于微调大语言模型

### LAION-400M 数据集

- 2022 年 3 月，全球最大的公开数据集 LAION-5B 发布

### SD UI 的发展

- SD V1 发布的同月，最著名的 SD web UI 开源发布
- 23 年 1 月，comfyUI 开源发布
- foocus UI 上线，专注提示词和图像，没有复杂参数调整，安装简单，显卡要求更低 4GB

[[Excalidraw/SD生态梳理.excalidraw\|SD生态梳理.excalidraw]]
