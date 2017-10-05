#!/bin/bash
sysdtime=`date +%Y%m%d-%H%M%S`
INPUT='../images/20170919/'
MODEL='./tmp/train/model/ckpt'
LOG='./tmp/train/log/'
TAR='./tmp/train/'

export CUDA_VISIBLE_DEVICES=0,1
cd /mnt/data/camus/project/
python3 ./cnn.py -train -maxstep 3000 -bsize 120 -input $INPUT -model $MODEL -log $LOG 1>model-$sysdtime.log 2>model-$sysdtime.error \
&& tar zcf model-$sysdtime.tar.gz $TAR \
&& rm -r $TAR*

exit 0
