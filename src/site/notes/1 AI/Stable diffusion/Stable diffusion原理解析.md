---
{"dg-publish":true,"permalink":"/1 AI/Stable diffusion/Stable diffusion原理解析/","noteIcon":""}
---


# LLM进化树

![1 AI/Stable diffusion/Fuw9fv9akAA_h0q(1) 3.jpg](/img/user/1%20AI/Stable%20diffusion/Fuw9fv9akAA_h0q(1)%203.jpg)
# 图像生成模型

- GAN 生成对抗模型， 生成模型VS判别模型
- Diffusion Model 

#todo #图像生成论文: 《what are diffusion models》论文详细阅读，介绍了更多的图像生成模型

# Diffusion models
## 两个过程

- 正向扩散：通过添加噪声来扰动数据分布
- 反向传播：将噪声从数据中去除，生成图像
#todo  #图像生成论文: 《Diffusion Models: A Comprehensive Survey of Methods and Applications》

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

#todo: Google deenlearningAI, 《How Diffusion Models Work》

# Stable Diffusion 发展线梳理
![1 AI/Stable diffusion/Pasted image 20250707155746.png](/img/user/1%20AI/Stable%20diffusion/Pasted%20image%2020250707155746.png)
## 主线

### 基础

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585)：
- [DDPM](https://arxiv.org/pdf/1907.05600v1.pdf)：15年提出的扩散概率模型，经过ddpm的改进之后，逐渐进入人们的视野。ddpm将“去噪”扩散概率模型应用到图像中，并且通过固定前向过程的方差、选择合适的反向过程参数化，以及使用高斯分布来表示条件概率，设计出了简单高效的扩散模型
- [SDM](https://arxiv.org/pdf/2006.11239v1.pdf)
- [Scope SDE](https://arxiv.org/pdf/2011.13456v1)
- DDPM、SDM、Scope SDE三大基础论文，奠定了扩散概率模型在图像生成领域的基础
	- 基于分数的生成模型SGM和扩散概率模型DDPM都可以视为由分数函数确定的随机微分方式Score SDE的离散化
	- 分数生成模型和扩散概率模型被连接到一个统一的框架中
	
### 高效采样
### 微调

## 辅线