# FINER

## Dependencies
- [Python 3.7.3](https://www.python.org/downloads/release/python-373/)</br>
- [TensorFlow 2.0.0](https://www.tensorflow.org/)</br>
- [NetworkX](https://github.com/haochenucr/networkx/tree/bugfix-for-to_scipy_sparse_matrix-function)</br>
Get the branch of NetworkX which fixed the bug of [#3985](https://github.com/networkx/networkx/pull/3985).</br>
- [NumPy](https://numpy.org/)</br>
- [SciPy](https://www.scipy.org/)</br>

## Get Started
### Data Preparation
- Unzip the file `./data.zip`.
- Run the script `./preprocessing/data_preprocess.sh` to preprocess the isoform sequence, domain and expression data, as well as the GO hierarchy. The ready-to-use data for model training will be save in the `./data/input/` directory.

### Train models
- Run the script `./src/train.sh` for training models. You can change the tissue index in the script to train models for different tissues appearing in the [tissue list]().
- Trained models will be saved in the `./saved_models/` directory. The model performance with predictions will be saved in the `./results/` directory.
