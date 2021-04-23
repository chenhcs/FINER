#!/bin/bash

python generate_go_hierarchy.py data
python seq_dataset_human.py data major
python seq_dataset_human.py data brain
python domain_dataset.py data major
python domain_dataset.py data brain
python construct_coexp_net.py data major
python construct_coexp_net.py data brain
