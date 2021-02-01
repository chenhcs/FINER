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
FINER is applied to predict tissue-specific isoform functions and interactions. The data used by FINER includes: gene-level functional annotation ground-truth, gene-level protein-protein interactions, isoform amino acid sequences, conserved domains of isoforms derived from their sequences, and isoform co-expression networks. Raw data are provided in the file `./data.zip`. Follow the following steps to prepare the tissue-specific datasets:
- Unzip the file `./data.zip`.
- Run the script to build co-expression networks of isoforms from their expression profiles in different RNA-seq experiments (measured in Transcripts Per Million or TPM), preprocess the isoform sequences, conserved domains for the use of the model, as well as build the Gene Ontology hierarchy. After which, the ready-to-use data for model training and evaluation will be save in the `./data/input/` directory.
```
sh ./preprocessing/data_preprocess.sh
```


## Training and evaluation
- Run the script for training models. Tissue index can be changed in the script to train models for different tissues appearing in the [tissue list](https://github.com/haochenucr/FINER/blob/main/src/train.sh).
```
sh ./src/train.sh
```
- Trained models will be saved in the `./saved_models/` directory. The model performance with predictions will be saved in the `./results/` directory.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
