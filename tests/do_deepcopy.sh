#!/bin/sh

apptag="[DO_DPCPY]"

# Check whether we are in correct dir.
if [ ! -f "./deepcopy_testdata.py" ]; then
    echo "$apptag ERROR: Please run this script from the '<repo_root>/tests/' directory."
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"

if [ -z "${SOURCE_DIR}" -o -z "${DEST_DIR}" ]; then
    echo "$apptag ERROR: SYNTAX ERROR ;)"
    echo "$apptag USAGE: $0 <source_dir> <dest_dir>"
    exit 1
fi

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "$apptag ERROR: Source directory '${SOURCE_DIR}' does not exist."
    exit 1
fi

if [ ! -d "${DEST_DIR}" ]; then
    echo "$apptag ERROR: Destination directory '${DEST_DIR}' does not exist."
    exit 1
fi


python ./deepcopy_testdata.py -s "${SOURCE_DIR}" -t "${DEST_DIR}" --file-list ./deepcopy_filelist.txt --not-so-deep --verbose