#!/bin/bash
sysdtime=`date +%Y%m%d-%H%M%S`
PWD='/mnt/data/camus/project/'
INPUT='../images/exp6/train'
MODEL='./tmp/train/model/exp6'
LOG='./tmp/train/log/'
TAR='./tmp/train/'
PREMOD='../images/exp1/model/final/07/model/exp1-2500'

export CUDA_VISIBLE_DEVICES=0
cd $PWD
python3 ./cnn.py -train -maxstep 1500 -bsize 320 -input $INPUT -model $MODEL -log $LOG -premod $PREMOD 1>model-$sysdtime.log 2>model-$sysdtime.error \
&& tar zcf model-$sysdtime.tar.gz $TAR \
&& rm -r $TAR*

exit 0
