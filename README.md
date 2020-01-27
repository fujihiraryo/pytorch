# [PyTorchの練習](https://pytorch.org/tutorials/)

## 環境構築(1回目)
```
conda create -n pytorch-tutorial -y
conda activate pytorch-tutorial
conda install pytorch torchvision -c pytorch -y
conda env export > environment.yml
```

## 環境構築(2回目以降)
```
conda env create -f environment.yml
conda activate pytorch-tutorial
```

## [image_classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

CIFAR10データセットを使った画像分類の練習\
データのロード
```
cd classifier
python dataloader.py
```
学習
```
python train.py
```
学習済みパラメータは`parameter/`に保存される。

テスト
```
python test.py
```
