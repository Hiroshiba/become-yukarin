# 使い方

## 必要なライブラリのインストール
```bash
pip install -r requirements.txt
```

## 学習させる
学習用のPythonスクリプトを実行するには、`become_yukarin`ライブラリをパス（PYTHONPATH）に通す必要があります。
例えば`scripts/extract_acoustic_feature.py`を以下のように書いて、パスを通しつつ実行します。

```bash
PYTHONPATH=`pwd` python scripts/extract_acoustic_feature.py ---
```

## 第１段階の学習
* 音声データを用意する
  * ２つのディレクトリに、入出力の音声データを置く（ファイル名を揃える）
* 音響特徴量切り出しをする
  * scripts/extract_acoustic_feature.py
* 学習を回す
  * train.py
* 実際に使用する
  * scripts/voice_conversion_test.py

## 第２段階の学習
* 音声データを用意する
  * １つのディレクトリの超大量の結月ゆかり音声データを置く
* 音響特徴量切り出しをする
  * scripts/extract_spectrogram_pair.py
* 学習を回す
  * train_sr.py
* 実際に使用する
  * scripts/super_resolution_test.py
* 実際に使う
  * SuperResolutionクラスとAcousticConverterクラスを使ってモデルを読み込ませればいい
  * [サンプルコード](https://github.com/Hiroshiba/become-yukarin/blob/ipynb/show%20vc%20and%20sr.ipynb)

## 参考
  * [ipynbブランチ](https://github.com/Hiroshiba/become-yukarin/tree/ipynb)に大量にサンプルが置いてある

## ファイル構造
```
├── become_yukarin  # このディレクトリは外から使えることを想定
│   ├── __init__.py
│   ├── config.py  # 学習の設定パラメータ
│   ├── data_struct.py  # データ構造の定義
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── dataset.py  # データ処理
│   │   └── utility.py
│   ├── model.py  # ニューラルネットワーク構造
│   ├── param.py  # 音声パラメータ
│   ├── updater.py  # chainerのUpdater
│   └── voice_changer.py  # 学習済みモデルを使って声質変換
├── recipe
│   ├── config.json  # 学習の設定パラメータ
│   └── recipe.json  # 複数の学習を回す時のパラメータ
├── requirements.txt  # 依存関係のあるライブラリ
├── scripts
│   ├── extract_acoustic_feature.py  # 音響特徴量切り出し
│   ├── launch.py  # 複数の学習を回す
│   ├── ln_apply_subset.py
│   ├── ln_atr503_to_subset.py
│   ├── ln_jnas_subset.py
├── tests
│   ├── __init__.py
│   └── test_dataset.py
└── train.py  # 学習用のスクリプト
```

## License
[MIT License](./LICENSE)
