#!/bin/sh


#datetag=$(date '+%Y-%m-%d_%H-%M-%S')
#logfile="log_meshlearn_lgi_${datetag}.txt"
./src/clients/meshlearn_lgi_train.py -v -w . tests/test_data/abide_lgi

