#!/bin/sh
hostname
source ~/.bashrc
micromamba activate gh-separation-test
cd /home/pmj0324/gh-separation-tranfer
python3 train.py ./config/config.yaml
