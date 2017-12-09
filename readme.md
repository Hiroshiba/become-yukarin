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
