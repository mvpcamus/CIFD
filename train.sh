#!/bin/bash
PWD='/mnt/data/camus/project/'
INPUT='../images/20171024.set/train'
MODEL='./tmp/train/model/ckpt'
LOG='./tmp/train/log/'
TAR='./tmp/train/'

export CUDA_VISIBLE_DEVICES=0
cd $PWD
rm $LOG*
rm $MODEL*
python3 ./cnn.py -train -maxstep 5000 -bsize 100 -input $INPUT -model $MODEL -log $LOG \
&& tar zcf train.tar.gz $TAR

exit 0
