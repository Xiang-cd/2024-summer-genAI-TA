# 扩散模型作业题
本仓库为清华大学2024年夏季生成式AI课程的作业题。

## 扩散模型损失函数的推导

扩散模型是一种生成模型的概率建模方式，因其表达能力强，易于训练，在很多生成任务都达到了非常优秀的水平。例如图像生成，视频生成，语音合成，三维数据生成，科学数据生成。扩散模型可以理解为一种可逆、基于分数的生成模型，其定义了前向过程(Forward Proces)和反向过程(Reverse Process)。

### 前向过程
其中前向过程是指不断向数据中添加随机噪声直到添加噪声后的数据分布十分接近于简单的高斯噪声分布。
具体而言，给定 $n$维度随机变量$x_0 \in \mathbb{R}^{n}$服从数据分布$q(x_0)$。扩散模型定义了一个从时间 $t = 0$ (干净数据)到 $t = T > 0$(带噪声数据)的前向马尔科夫扩散过程。
前向过程(包括$x_0$)的联合分布可以表示为:

$$
q(x_{0:T}) = q(x_0) \prod_{t=1}^T q(x_t \mid x_{t-1})
$$

其中 $q(x_t \mid x_{(t-1)})$ 为前向扩散核，其被定义为 $\mathcal{N}\left(\mathbf{x}_t \text{;} \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$ ，其中 $\beta_t$ 为预先定义好的方差序列 $0<\beta_1 … \beta_T<1$ 中 $t$ 时刻对应的值。
由于高斯分布的良好性质($\mathcal{N}\left(0, \sigma_1^2 \mathbf{I}\right)+\mathcal{N}\left(0, \sigma_2^2 \mathbf{I}\right) \sim \mathcal{N}\left(0,\left(\sigma_1^2+\sigma_2^2\right) \mathbf{I}\right)$)，
对此可以非常容易地在前向过程中采样任意 $t$ 时刻的分布(记 $\alpha_t:=1-\beta_t$ 并记 $\bar{\alpha}_t:=\prod_{k=1}^t \alpha_k$ ):

$$
q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)
$$

通过合理的方差序列选择，可以使得 $q\left(\boldsymbol{x}_T\right) \approx \mathcal{N}\left(\boldsymbol{x}_T \mid \mathbf{0},   \boldsymbol{I}\right)$，
并且前向过程中信噪比(Signal-to-Noise Ratio，SNR) $\frac{\bar{\alpha}_t }{ (1-\bar{\alpha})^2}$ 递减。


### 反向过程
扩散模型的反向过程是指从简单高斯噪声分布开始，逐渐去除噪声，从而得到数据采样的样本。
其中去噪过程由一组参数$\theta$定义，最终的采样和由参数$\theta$定义的数据分布可以表述为 $p_\theta\left(x_0\right):=\int p_\theta\left(x_{0: T}\right) d x_{1: T}$ 。
具体的马尔科夫过程从 $p(x_T) = \mathcal{N}(x_T; 0, \mathbf{I})$ 开始，反向过程中的联合分布可以表示为:


$$
p_\theta\left(\mathbf{x}_{0: T}\right):=p\left(\mathbf{x}_T\right) \prod^{T}_{t=1} p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)
$$

其中 $p(x_{t-1} \mid x_t)$ 为反向扩散核，其被定义为由参数$\theta$根据当前噪声数据和时间$t$预测的为参数的高斯分布 $\mathcal{N}\left(\mathbf{x}_{t-1} \text{;} \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \boldsymbol{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right)$。

请参考[论文1](https://arxiv.org/abs/2006.11239), [论文2](https://arxiv.org/abs/1503.03585)，写出扩散模型在噪声预测重参数化情况下的损失函数以及其推导过程。



## 扩散模型损失函数实现
我们已经在理论上推导了扩散模型的损失函数，下面我们将其在代码层面进行实现。

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
