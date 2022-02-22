#!/bin/sh

SOURCE_DIR="$1"
DEST_DIR="$2"

if [ -z "${SOURCE_DIR}" -o -z "${DEST_DIR}" ]; then
    echo "SYNTAX ERROR ;)"
    echo "USAGE: $0 <source_dir> <dest_dir>"
    exit 1
fi

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "Source directory '${SOURCE_DIR}' does not exist."
fi

if [ ! -d "${DEST_DIR}" ]; then
    echo "Destination directory '${DEST_DIR}' does not exist."
fi

if [ ! -f "./deepcopy_testdata.py" ]; then
    echo "Please run this from the <repo_root>/tests/ directory."
fi

python ./deepcopy_testdata.py -s "${SOURCE_DIR}" -t "{$DEST_DIR}" -f ./deepcopy_filelist.txt