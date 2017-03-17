#!/bin/bash

USAGE="Usage: command <net in matconvnet format> <directory with test data> <output directory>"

if [ $# == 0 ] ; then
    echo $USAGE
    exit 1;
fi

command -v matlab >/dev/null 2>&1 || { echo >&2 "Matlab is not installed.  Aborting."; exit 1; }
scrpits_dir="$(dirname "${BASH_SOURCE[0]}")"
matlab_command="addpath('$scrpits_dir'); prepare_test_data_for_net('$1', '$2', '$3'); quit;"
echo "$matlab_command"
matlab -nodesktop -nojvm -r "$matlab_command"
