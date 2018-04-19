#!/bin/bash
PWD='/mnt/data/camus/project/'
INPUT='../images/exp6/train'
MODEL='./tmp/train/model/exp6'
LOG='./tmp/train/log/'
TAR='./tmp/train/'
PREMOD='../images/exp1/model/final/07/model/exp1-2500'

export CUDA_VISIBLE_DEVICES=0
cd $PWD
rm $LOG*
rm $MODEL*
python3 ./cnn.py -train -maxstep 1000 -bsize 160 -input $INPUT -model $MODEL -log $LOG -premod $PREMOD \
&& tar zcf train.tar.gz $TAR

exit 0
