#!/bin/bash

SRC_DIR="auto_grader"
DEFAULT_ENTRY="main.py"
TEST_SET_ENTRY="synthetic_data_generator/synthetic_data_generator.py"

# Check if the --testset argument is provided
if [ "$1" == "--testset" ]; then
    python $TEST_SET_ENTRY
else
    python $SRC_DIR/$ENTRY_FILE
fi

