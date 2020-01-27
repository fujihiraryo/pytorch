# [PyTorch の練習](https://pytorch.org/tutorials/)

## 環境構築(1 回目)

```
conda create -n pytorch-tutorial -y
conda activate pytorch-tutorial
conda install pytorch torchvision -c pytorch -y
conda env export > environment.yml
```

## 環境構築(2 回目以降)

```
conda env create -f environment.yml
conda activate pytorch-tutorial
```
