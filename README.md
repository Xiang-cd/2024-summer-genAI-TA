# 扩散模型作业题


## 扩散模型损失函数的推导




## 扩散模型损失函数实现

### 环境设置
安装conda
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
mkdir ./miniconda3
bash /tmp/miniconda.sh -b -u -p ./miniconda3
./miniconda3/bin/conda init bash
```
克隆相关仓库以及安装相关依赖
```sh
git clone git@github.com:Xiang-cd/2024-summer-genAI-TA.git
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

跑通训练代码
```sh
python main.py --config cifar10.yml --exp cf10self --doc logcf10self --ni
```

