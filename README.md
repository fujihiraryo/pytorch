# [PyTorch の練習](https://pytorch.org/tutorials/)

## 環境構築(1 回目)

```
conda create -n pytorch-tutorial python=3.6 -y
conda activate pytorch-tutorial
conda install pytorch torchvision -c pytorch -y
conda env export > environment.yml
```

## 環境構築(2 回目以降)

```
conda env create -f environment.yml
conda activate pytorch-tutorial
```

## 環境作り直し

```
conda deactivate
conda remove -n pytorch-tutorial --all -y
```
