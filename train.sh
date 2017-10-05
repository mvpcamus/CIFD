#!/bin/bash
PWD='/mnt/data/camus/project/'
INPUT='../images/20170919/'
MODEL='./tmp/train/model/ckpt'
LOG='./tmp/train/log/'
TAR='./tmp/train/'

export CUDA_VISIBLE_DEVICES=1
cd $PWD
python3 ./cnn.py -train -maxstep 3000 -bsize 120 -input $INPUT -model $MODEL -log $LOG \
&& tar zcf train.tar.gz $TAR

exit 0
