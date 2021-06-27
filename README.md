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
- [scikit-learn >= 0.21.3](https://scikit-learn.org/stable/)</br>

FINER is implemented using TensorFlow. We suggest users to install the TensorFlow environment with GPU support using [Anaconda](https://anaconda.org/anaconda/tensorflow-gpu).
```
conda create -n tensorflow-gpu
conda activate tensorflow-gpu
conda install -c anaconda tensorflow-gpu
```

## Data preparation
FINER is applied to predict tissue-specific isoform functions and interactions. The data used by FINER includes: (i) gene-level functional annotation ground-truth, (ii) gene-level protein-protein interactions, (iii) isoform amino acid sequences, (iv) conserved domains of isoforms derived from their sequences, (v) and isoform co-expression networks. Raw data are provided in the file `./data.tar.gz`. Follow the steps to prepare the tissue-specific datasets:
- Extract the contents of the `./data.tar.gz` file.
```
tar -zxvf data.tar.gz
```
- Run the `data_preprocess.sh` script to build co-expression networks of isoforms from their expression profiles in different RNA-seq experiments (measured in Transcripts Per Million or TPM), convert the isoform sequences, conserved domains to Numpy arrays for the use of the model, as well as build the Gene Ontology hierarchy. After which, the ready-to-use data for model training and evaluation will be save in the `./data/input/` directory.
```
cd preprocessing/
sh data_preprocess.sh
```


## Training and evaluation
We train each FINER model with one Nvidia K80 GPU which significantly accelerated the training process than CPUs.
- Run the script for training models. Tissue index can be changed in the script to train models for different tissues appearing in the [list](https://github.com/haochenucr/FINER/blob/main/src/train.sh).
```
cd src/
sh train.sh
```
- Trained models will be saved in the `./saved_models/` directory. The model performance with predictions will be saved in the `./results/` directory.
- Modify the files in the `./hyper_prms/` directory to adjust model hyper-parameters.

## Custom tissue-specific datasets
- Create a new folder e.g. `./my_data/`.
- Put your raw data in the folder created above in the format as specified in `./data_format/`.
- Run the following command to preprocess the data.
```
cd preprocessing/
sh custom_dataset.sh {your dataset}
```
- Create a corresponding file of hyper-parameters in the `./hyper_prms/` directory.
- Run the following command to train a model on the dataset for the specific tissue of interest.
```
cd src/
python joint_train.py {your dataset} {tissue ID}
```

## Predictions and performance of FINER and the other methods compared
In the `./preds_and_perf/` directory, you can find:
- Predictions on tissue-specific GO terms and tissue-specific isoform-isoform interactions by FINER.
- Predicted tissue-specific isoform-isoform interaction networks of [TENSION](https://www.nature.com/articles/s41598-019-50119-x).
- Prediction performance on each tissue-specific GO term of FINER and the other methods compared in the paper.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Email: hchen069@ucr.edu
