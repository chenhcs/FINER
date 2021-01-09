#!/bin/bash

python generate_go_hierarchy.py
python seq_dataset_human.py major
python seq_dataset_human.py brain
python domain_dataset.py major
python domain_dataset.py brain
python construct_coexp_net.py
