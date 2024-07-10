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

where $p(x_{t-1} \mid x_t)$ is the back diffusion kernel, which is defined as the Gaussian distribution $\mathcal{N}\left(\mathbf{x}_{t-1} \text{;} \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \boldsymbol{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right)$ with parameters $\theta$ predicted from the current noise data and time $t$.

refer to [paper1](https://arxiv.org/abs/2006.11239), [paper2](https://arxiv.org/abs/1503.03585)，Write the loss function of the diffusion model in the case of noise prediction reparameterization and its derivation process.



## Diffusion model loss function implementation
We have theoretically derived the loss function of the diffusion model. Now let’s implement it at the code level.

### evn setup
1. install conda
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
mkdir ./miniconda3
bash /tmp/miniconda.sh -b -u -p ./miniconda3
./miniconda3/bin/conda init bash
# restart terminal after conda init
```
2. clone the repository and install requirements
```sh
git clone https://github.com/Xiang-cd/2024-summer-genAI-TA.git
cd 2024-summer-genAI-TA
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
```
3. download cifar10 dataset
the container already has the cifar10 dataset at `/data/imagenet`, copy the tar file to the path the torchvision going to download the dataset or download it manually.
```
mkdir -p cf10self/datasets/cifar10
cp /data/imagenet/cifar-10-python.tar.gz cf10self/datasets/cifar10
mkdir -p cf10self/datasets/cifar10_test
cp /data/imagenet/cifar-10-python.tar.gz cf10self/datasets/cifar10_test
```

4. run the training code (better run in tmux)
```sh
python main.py --config cifar10.yml --exp cf10self --doc logcf10self --ni
```

5. run the evaluation code
```sh
python main.py --config cifar10.yml --exp cf10 --doc logcf10 --ni --sample --timesteps 100 --fid
```

## Open problem (Optional)

### Problem
Using stable diffusion as an example, compare two popular sampling acceleration methods in terms of effectiveness, time, and memory requirements: training-free ODE solvers (e.g., DPM-solver) and distillation-based methods (e.g., LCM). The test prompt list can be found in prompt.jsonl.

### Evaluation metrics
* Effectiveness: human preference, which images generated by the two methods do you prefer?
* Inference Time: time required to generate a 512x512 image
* Memory: compare the memory requirements for inference and training for both methods。
