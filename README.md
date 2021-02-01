# FINER
FINER is a unified deep learning framework to jointly predict protein isoform functions and isoform-isoform interactions.

## Dependencies
- [Python >= 3.7.3](https://www.python.org/downloads/release/python-373/)</br>
- [TensorFlow >= 2.0.0](https://www.tensorflow.org/)</br>
- [Keras >= 2.3.1](https://keras.io/)</br>
- [NetworkX](https://github.com/haochenucr/networkx/tree/bugfix-for-to_scipy_sparse_matrix-function)</br>
Get the branch of NetworkX which fixed the bug of [#3985](https://github.com/networkx/networkx/pull/3985).</br>
- [NumPy >= 1.17.3](https://numpy.org/)</br>
- [SciPy >= 1.3.2](https://www.scipy.org/)</br>

FINER is implemented using TensorFlow. We suggest users to install the TensorFlow environment with GPU support using [Anaconda](https://anaconda.org/anaconda/tensorflow-gpu).
```
conda create -n tensorflow-gpu
conda activate tensorflow-gpu
conda install -c anaconda tensorflow-gpu
```

## Data preparation
- Unzip the file `./data.zip`.
- Run the script `./preprocessing/data_preprocess.sh` to preprocess the isoform sequence, domain and expression data, as well as the GO hierarchy. The ready-to-use data for model training will be save in the `./data/input/` directory.

## Training and evaluation
- Run the script `./src/train.sh` for training models. You can change the tissue index in the script to train models for different tissues appearing in the [tissue list]().
- Trained models will be saved in the `./saved_models/` directory. The model performance with predictions will be saved in the `./results/` directory.

## Predicted tissue-specific isoform functions
## Predicted tissue-specific isoform-isoform interaction networks
