#!/bin/bash

SRC_DIR="auto_grader"
DEFAULT_ENTRY="main.py"
TEST_SET_ENTRY="synthetic_data_generator/synthetic_data_generator.py"
TRULENS_ENTRY="trulens/eval.py"

# Check if the --testset or --trulens argument is provided
if [ "$1" == "--testset" ]; then
    python $TEST_SET_ENTRY
elif [ "$1" == "--trulens" ]; then
    python $TRULENS_ENTRY
else
    python $SRC_DIR/$DEFAULT_ENTRY
fi