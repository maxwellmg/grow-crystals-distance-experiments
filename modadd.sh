#!/bin/bash

echo "My task ID:" $LLSUB_RANK
echo "Number of Tasks:" $LLSUB_SIZE

nohup python modadd.py $LLSUB_RANK $LLSUB_SIZE

