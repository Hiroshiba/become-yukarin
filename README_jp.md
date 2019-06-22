# Become Yukarin: 誰でも好きなキャラの声に
Become Yukarinは、機械学習（ディープラーニング）で声質変換を実現するリポジトリです。
元の声と好きな声の音声データを大量に用いて機械学習することで、
元の声を好きな声に変換することができるようになります。

[English README](./README.md)

## 推奨環境
* Linux OS
* Python 3.6

## 準備
```bash
# 必要なライブラリをインストール
pip install -r requirements.txt
```

## 学習させる
学習用のPythonスクリプトを実行するには、`become_yukarin`ライブラリをパス（PYTHONPATH）に通す必要があります。
例えば`scripts/extract_acoustic_feature.py`を以下のように書いて、パスを通しつつ実行します。

```bash
PYTHONPATH=`pwd` python scripts/extract_acoustic_feature.py ---
```

### 第１段階の学習
* 音声データを用意する
  * ２つのディレクトリに、入出力の音声データを置く（ファイル名を揃える）
* 音響特徴量を作成する
  * `scripts/extract_acoustic_feature.py`
* 学習を回す
  * `train.py`
* テストする
  * `scripts/voice_conversion_test.py`

### 第２段階の学習
* 音声データを用意する
  * １つのディレクトリに音声データを置く
* 音響特徴量を作成する
  * `scripts/extract_spectrogram_pair.py`
* 学習を回す
  * `train_sr.py`
* テストする
  * `scripts/super_resolution_test.py`
* 別の音声データを変換する
  * SuperResolutionクラスとAcousticConverterクラスを使うことで変換できます
  * [サンプルコード](https://github.com/Hiroshiba/become-yukarin/blob/ipynb/show%20vc%20and%20sr.ipynb)

## 参考
  * [ipynbブランチ](https://github.com/Hiroshiba/become-yukarin/tree/ipynb)に大量にサンプルが置いてあります
  * [解説ブログ](https://hiroshiba.github.io/blog/became-yuduki-yukari-with-deep-learning-power/)
  * [Realtime Yukarin](https://github.com/Hiroshiba/realtime-yukarin)を使うことで、リアルタイムに声質変換することができます

## License
[MIT License](./LICENSE)
