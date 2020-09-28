# PyTorch の練習リポジトリ

## 環境構築(1 回目)

```text
conda create -n pytorch-tutorial python=3.6 -y
conda activate pytorch-tutorial
conda install pytorch torchvision -c pytorch -y
conda install flake8 -y
conda install pytest -y
conda env export > environment.yml
```

## 環境構築(2 回目以降)

```text
conda env create -f environment.yml
conda activate pytorch-tutorial
```

## 環境削除

```text
conda deactivate
conda remove -n pytorch-tutorial --all -y
```
