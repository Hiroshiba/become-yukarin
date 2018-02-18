from setuptools import setup, find_packages

setup(
    name='become_yukarin',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/become-yukarin',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    description='become Yuduki Yukari with DeepLearning power.',
    license='MIT License',
    install_requires=[
        'numpy',
        'chainer',
        'librosa',
        'pysptk',
        'pyworld',
        'fastdtw',
        'nnmnkwii',
        'chainerui',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
    ]
)
