#!/bin/bash
PWD='/mnt/data/camus/project/'
INPUT='../images/20170830/'
MODEL='./tmp/test/model/ckpt'
LOG='./tmp/test/log/'

export CUDA_VISIBLE_DEVICES=1
cd $PWD
rm $LOG*
python3 ./cnn.py -bsize 100 -input $INPUT -model $MODEL -log $LOG -fmap show

exit 0
