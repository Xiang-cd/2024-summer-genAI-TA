# assignments of diffusion models

This repository contains the homework questions for Tsinghua University's 2024 Summer Generative AI course.

## Derivation of the Diffusion Model Loss Function

The diffusion model is a probabilistic modeling method for generative models. Due to its strong expressiveness and ease of training, it has achieved excellent results in many generative tasks, such as image generation, video generation, speech synthesis, 3D data generation, and scientific data generation. The diffusion model can be understood as a reversible, score-based generative model that defines the forward process and the reverse process.

### forward process
The forward process refers to continuously adding random noise to the data until the data distribution after adding noise is very close to the simple Gaussian noise distribution.
Specifically, given an $n$-dimensional random variable $x_0 \in \mathbb{R}^{n}$ that obeys the data distribution $q(x_0)$. The diffusion model defines a forward Markov diffusion process from time $t = 0$ (clean data) to $t = T > 0$ (noisy data).
The joint distribution of the forward process (including $x_0$) can be expressed as:

$$
q(x_{0:T}) = q(x_0) \prod_{t=1}^T q(x_t \mid x_{t-1})
$$

Where $q(x_t \mid x_{(t-1)})$ is the forward diffusion kernel, which is defined as $\mathcal{N}\left(\mathbf{x}_t \text{;} \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$ , where $\beta_t$ is the value corresponding to the $t$ moment in the predefined variance sequence $0<\beta_1 … \beta_T<1$ .
Due to the good properties of Gaussian distribution ($\mathcal{N}\left(0, \sigma_1^2 \mathbf{I}\right)+\mathcal{N}\left(0, \sigma_2^2 \mathbf{I}\right) \sim \mathcal{N}\left(0,\left(\sigma_1^2+\sigma_2^2\right) \mathbf{I}\right)$),
it is very easy to sample the distribution at any $t$ time in the forward process (note $\alpha_t:=1-\beta_t$ and $\bar{\alpha}_t:=\prod_{k=1}^t \alpha_k$):

$$
q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)
$$

By choosing a reasonable variance sequence, we can make $q\left(\boldsymbol{x}_T\right) \approx \mathcal{N}\left(\boldsymbol{x}_T \mid \mathbf{0}, \boldsymbol{I}\right)$, and the signal-to-noise ratio (SNR) $\frac{\bar{\alpha}_t }{ (1-\bar{\alpha})^2}$ decrease in the forward process.


### reverse process
The reverse process of the diffusion model refers to starting from a simple Gaussian noise distribution and gradually removing the noise to obtain a sample of data sampling.
The denoising process is defined by a set of parameters $\theta$. The final sampling and the data distribution defined by the parameters $\theta$ can be expressed as $p_\theta\left(x_0\right):=\int p_\theta\left(x_{0: T}\right) d x_{1: T}$.
The specific Markov process starts from $p(x_T) = \mathcal{N}(x_T; 0, \mathbf{I})$. The joint distribution in the reverse process can be expressed as:


$$
p_\theta\left(\mathbf{x}_{0: T}\right):=p\left(\mathbf{x}_T\right) \prod^{T}_{t=1} p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)
$$

其中 $p(x_{t-1} \mid x_t)$ 为反向扩散核，其被定义为由参数$\theta$根据当前噪声数据和时间$t$预测的为参数的高斯分布 $\mathcal{N}\left(\mathbf{x}_{t-1} \text{;} \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \boldsymbol{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right)$。

refer to [paper1](https://arxiv.org/abs/2006.11239), [paper2](https://arxiv.org/abs/1503.03585)，Write the loss function of the diffusion model in the case of noise prediction reparameterization and its derivation process.



## Diffusion model loss function implementation
We have theoretically derived the loss function of the diffusion model. Now let’s implement it at the code level.

### 环境设置
1. 安装conda
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
mkdir ./miniconda3
bash /tmp/miniconda.sh -b -u -p ./miniconda3
./miniconda3/bin/conda init bash
```
2. 克隆相关仓库以及安装相关依赖
```sh
git clone git@github.com:Xiang-cd/2024-summer-genAI-TA.git
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
```
3. 下载cifar10数据集
学期提供的容器中`/data/imagenet`目录下已经有了cifar10数据集, 将对应的数据集拷贝到torchvision将要下载到的路径即可使用, 同学们也可以自行下载和指定。

4. 跑通训练代码
```sh
python main.py --config cifar10.yml --exp cf10self --doc logcf10self --ni
```

5. 进行采样测试
```sh
python main.py --config cifar10.yml --exp cf10 --doc logcf10 --ni --sample --timesteps 100 --fid
```

## 开放题
开放题旨在引入扩散生成模型领域中比较关键的问题以及现有的相关工作, 激发同学们思考。

### 扩散模型生成的崩坏问题
扩散模型生成的图像存在人脸, 人手的崩坏问题, 以及不一致等问题。
例如[Sora生成问题](https://www.reddit.com/r/OpenAI/comments/1arrqpz/funny_glitch_with_sora_interesting_how_it_looks/)。
如何解决扩散模型生成的崩坏问题是一项重要的研究方向, 也是困扰工业界的重要问题。一部分的研究工作显示, 增大模型能力能够缓解部分问题。
[基于人类偏好的方法](https://arxiv.org/abs/2311.12908)也能够一定程度上缓解这一问题。
做到稳定鲁棒的缓解崩坏将带来巨大的商业价值和学术影响力。
对此, 同学们可以考虑在[stable diffusion](https://github.com/CompVis/stable-diffusion)进行采样, 得到崩坏样本, 研究崩坏出现的规律, 提出解决方案。


### 扩散模型采样加速
扩散模型需要通过多步的迭代来生成, 上千步的迭代使得推理成本十分昂贵, 如何加速扩散模型的采样是重要的研究议题。
[DPM-solver](https://arxiv.org/abs/2206.00927)是一种广泛采用的采样加速方法, 能够做到20步左右生成高质量样本, 但依然难做到单步采样。
基于蒸馏的方法例如[CM](https://github.com/openai/consistency_models?tab=readme-ov-file), [LCM](https://arxiv.org/abs/2310.04378), 通过训练的方法来加速采样。
同学们可以阅读相关论文, 尝试运行相关的代码, 并体会不同加速方法的效果。
