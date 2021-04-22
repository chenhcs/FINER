#!/bin/bash
folder=$1

python generate_go_hierarchy.py $folder
python seq_dataset_human.py $folder major
python seq_dataset_human.py $folder brain
python domain_dataset.py $folder major
python domain_dataset.py $folder brain
python construct_coexp_net.py $folder major
python construct_coexp_net.py $folder brain
