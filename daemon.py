from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import time
from image import ImageGen
import cnn

class Daemon(object):
  def __init__(self):
    input_path = '/mnt/nas/temp/'
    default_file = 'data.csv'
    self.png_path = '/home/camus/data/project/demo/input/'
    self.model_path = '/home/camus/data/project/tmp/test/model/v0.4/20171103-5k'
    self.fmap_dir = '/home/camus/data/project/demo/output/'
    self.messages = ['Normal condition', 'Stator fault detected', 'Rotor fault detected', 'Bearing fault detected']
    self.eventime = None
    self.result = None
    self.prob = []

    if os.listdir(input_path):
      self.input_file = input_path + os.listdir(input_path)[-1]
    else:
      self.input_file = input_path + default_file

    print('--------------------------------------------------------')
    print('Induction motor fault detector')
    print('    looking at '+input_path)
    print('--------------------------------------------------------')


  def loop(self):
    if os.path.isfile(self.input_file):
      try:
        next_ = int(self.input_file.split('(')[1].split(')')[0]) + 1
        check_file = self.input_file.split('(')[0] + '(' + str(next_) + ').csv'
      except:
        check_file = self.input_file.replace('.csv', ' (2).csv')

      if os.path.isfile(check_file):
        lines = subprocess.check_output(['wc','-l',self.input_file]).decode()
        lines = int(lines.split(' ')[0])
        if int(lines) >= 1000003:
          self.eventime = time.localtime()
          print('[{}] a new raw data detected.'.format(time.strftime('%Y-%m-%d %H:%M:%S', self.eventime)))
          print('    {}'.format(self.input_file))
          png_file = self.png_path+time.strftime('%Y%m%d-%H%M%S',self.eventime)+'.png'
          ImageGen().run(self.input_file, png_file)
          print('[{}] heat map image generated.'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
          print('    {}'.format(png_file))
          self.result, self.prob = cnn.do_infer(png_file, self.model_path, self.fmap_dir)
        self.input_file = check_file
        return self.result, max(self.prob), png_file
      else:
        return None

    else:
      return None

  def complete(self):
        print('[{}] inference completed.'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        print('    {} at {}'.format(self.messages[self.result],
                                    time.strftime('%Y.%m.%d.(%a) %H:%M:%S',self.eventime)))
        print('     - Normal       : {:10.6f} %'.format(self.prob[0]*100))
        print('     - Stator Fault : {:10.6f} %'.format(self.prob[1]*100))
        print('     - Rotor Fault  : {:10.6f} %'.format(self.prob[2]*100))
        print('     - Bearing Fault: {:10.6f} %'.format(self.prob[3]*100))
        print('--------------------------------------------------------')

if __name__ == '__main__':
  daemon = Daemon()
  while True:
    if daemon.loop() is not None: daemon.complete()
    time.sleep(0.1)
