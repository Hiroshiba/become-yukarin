# Become Yukarin: Convert your voice to favorite voice
Become Yukarin is a repository for voice conversion with a Deep Learning model.
By traingin with a large amount of the original and favorite voice,
The Deep Learning model can convert the original voice to the favorite voice.

[Japanese README](./README_jp.md)

## Supported environment
* Linux OS
* Python 3.6

## Preparation
```bash
# install required libraries
pip install -r requirements.txt
```

## Training
To run a Python script for training,
you should set the environment variable `PYTHONPATH` to find the `become_yukarin` library.
For example, you can execute `scripts/extract_acoustic_feature.py` with the following command:

```bash
PYTHONPATH=`pwd` python scripts/extract_acoustic_feature.py ---
```

## First Stage Model
* Prepare voice data
  * Put input/target voice data in two directories (with same file names)
* Create acoustic feature
  * `scripts/extract_acoustic_feature.py`
* Train
  * `train.py`
* Test
  * `scripts/voice_conversion_test.py`

## Second Stage Model
* Prepare voice data
  * Put input/target voice data in two directories
* Create acoustic feature
  * `scripts/extract_spectrogram_pair.py`
* Train
  * `train_sr.py`
* Test
  * `scripts/super_resolution_test.py`
* Convert other voice data
  * Use SuperResolution class and AcousticConverter class
  * [sample code](https://github.com/Hiroshiba/become-yukarin/blob/ipynb/show%20vc%20and%20sr.ipynb)

## Reference
  * [ipynb branch](https://github.com/Hiroshiba/become-yukarin/tree/ipynb): Other sample code
  * [Commentary Blog (Japanese)](https://hiroshiba.github.io/blog/became-yuduki-yukari-with-deep-learning-power/)
  * [Realtime Yukarin](https://github.com/Hiroshiba/realtime-yukarin): Real-time voice conversion system

## License
[MIT License](./LICENSE)
