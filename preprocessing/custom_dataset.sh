#!/bin/bash
folder=$1

python generate_go_hierarchy.py $folder
python seq_dataset_human.py $folder
python domain_dataset.py $folder
python construct_coexp_net.py $folder
