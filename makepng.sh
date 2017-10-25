#!/bin/bash
INPUT='/mnt/nas/data/waveforms/20171024/25Hz-50%/25Hz-50%-'
OUTPUT='/home/camus/data/images/20171024/25Hz-50%-'
START=1
END=300

python3 ./image.py $INPUT $OUTPUT $START $END

exit 0
