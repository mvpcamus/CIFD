from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from image import ImageGen

img = ImageGen()
input_ = '/home/camus/data/waveforms/20170919/25Hz-50%-'
output = '/home/camus/data/images/20170919/25Hz-50%-'

for i in range(301):
  number = str(i+1).zfill(3)
  try:
    img.run(input_+number+'.csv', output+number+'.png')
  except:
    continue
  print('processed: '+input_+number+'.csv')
