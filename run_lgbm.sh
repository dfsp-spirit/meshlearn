#!/bin/sh

#abide_lgi_dir="tests/test_data/abide"
#if [ ! -d "${abide_lgi_dir}" ]; then
#    echo "ERROR: Cannot find the ABIDE lGI data at '${abide_lgi_dir}'."
#    echo "  Please run the 'tests/do_deepcopy.sh' on your recon-all"
#    echo "  ABIDE output directory after computing pial-lgi in FreeSurfer."
#    exit 1
#fi

#datetag=$(date '+%Y-%m-%d_%H-%M-%S')
#logfile="log_meshlearn_lgi_${datetag}.txt"
python ./src/meshlearn/clients/meshlearn_lgi.py -v

