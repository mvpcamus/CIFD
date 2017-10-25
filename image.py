from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import csv
import copy
import six
import matplotlib.pyplot as plt
import matplotlib.image as im

class ImageGen(object):

  def __init__(self):
    self.index = ['Normal', 'Stator fault', 'Rotor fault', 'Bearing fault']


  def run(self, csv, png=None, size=1000, human=False):
    '''
    Draw or save correlation matrix graph images
    args:
      csv = string of input CSV filepath
      png = string of output image filepath
      size = size of graph axes
      human = True: graph for human eyes, False: for machine learning
    '''
    rawdata = self._readCsv(csv)
    images = self._drawGraph(rawdata, size, human)

    if png:
      png = png.replace('.png','')
      if len(rawdata)==1:
        im.imsave('%s.png'%png, images[self.index[0]])
      else:
        for i in range(len(rawdata)):
          im.imsave('%s-%d.png'%(png,i), images[self.index[i]])
    else:
      for i in range(len(rawdata)):
        plt.subplot(1,len(self.index),i+1)
        plt.imshow(images[self.index[i]])
        plt.title(self.index[i])
      plt.show()

    return


  def _readCsv(self, csvpath):
    '''
    read in csv input files and generate rawdata dict.
    args:
      csvpath = string of input CSV filepath
    return:
      rawdata = dictionary of each index dictionary (see self.index)
                each index dictionary consists of
                {'c': np.array consists of [ current 1,  current 2 ]
                 'v': np.array consists of [ voltage 1,  voltage 2 ]
                 'm': np.array consists of [ movement 1, movement 2 ]}
    '''
    rawdata = {}

    # read in normal data csv file if exists
    if os.path.isfile(csvpath):
      with open(csvpath) as csvfile:
        rawdata[self.index[0]] = {'c':[], 'v':[], 'm':[]}
        reader = csv.reader(csvfile, delimiter=',')
        six.next(reader)
        six.next(reader)
        six.next(reader)  # skip three lines of index in csv file
        for row in reader:
          values = []
          for col in row:
            try:
              values.append(float(col))
            except:
              print('ERROR: inadequate data value: ', col)
              print('       in file: ', aInput)
              exit()
          rawdata[self.index[0]]['c'].append([ values[1], values[2] ])
          rawdata[self.index[0]]['v'].append([ values[3], values[4] ])
          rawdata[self.index[0]]['m'].append([ values[5], values[6] ])
        if rawdata[self.index[0]]['c'] == []:
          print('ERROR: cannot find data in file: ', aInput+'.csv')
          exit()
        else:
          rawdata[self.index[0]]['c'] = np.array(rawdata[self.index[0]]['c'])
          rawdata[self.index[0]]['v'] = np.array(rawdata[self.index[0]]['v'])
          rawdata[self.index[0]]['m'] = np.array(rawdata[self.index[0]]['m'])
    else:
      print('ERROR: cannot find input file: ', csvpath)
      exit()

    # read in fault data csv file if exists
    faultpath = csvpath.replace('.csv','_fault.csv')
    if os.path.isfile(faultpath):
      with open(faultpath) as csvfile:
        for i in range(1,4):
          rawdata[self.index[i]] = {'c':[], 'v':[], 'm':[]}
        reader = csv.reader(csvfile, delimiter=',')
        six.next(reader)  # skip a line of index in csv file
        for row in reader:
          values = []
          for col in row:
            try:
              values.append(float(col))
            except:
              print('ERROR: inadequate fault value: ', col)
              print('       in file: ', faultpath)
              exit()
          rawdata[self.index[1]]['c'].append([ values[3], values[4] ])
          rawdata[self.index[2]]['c'].append([ values[5], values[6] ])
          rawdata[self.index[3]]['c'].append([ values[7], values[8] ])
          if len(values) == 11:
            rawdata[self.index[3]]['m'].append([ values[9], values[10] ])
        if not rawdata[self.index[1]]['c']:
          print('ERROR: cannot find data in file: ', faultpath)
          exit()
        else:
          for cond in list(rawdata.keys()):
            if cond == self.index[0]:
              pass
            else:
              rawdata[cond]['c'] = np.array(rawdata[cond]['c'])
              rawdata[cond]['v'] = np.array(rawdata[cond]['v']) if rawdata[cond]['v'] \
                  else copy.deepcopy(rawdata[self.index[0]]['v'])
              rawdata[cond]['m'] = np.array(rawdata[cond]['m']) if rawdata[cond]['m'] \
                  else copy.deepcopy(rawdata[self.index[0]]['m'])

    return rawdata


  def _drawGraph(self, rawdata, size, human):
    '''
    generate each graph of cvm (current, voltage, movement) correlation matrix
    args:
      rawdata = rawdata container (dictionary)
      size = size of graph axes
      human = True: graph for human eyes, False: for machine learning
    return:
      images = dictionary of graphs for each index (see self.index)
    '''
    images = {}
    gmin = {'c':0, 'v':0, 'm':0}
    gmax = {'c':0, 'v':0, 'm':0}

    # calculate min and max of each cvm for graph scaling
    for cvm in ['c','v','m']:
      for cond in list(rawdata.keys()):
        gmin[cvm] = min( gmin[cvm], rawdata[cond][cvm].min() )
        gmax[cvm] = max( gmax[cvm], rawdata[cond][cvm].max() )

    # data normalization and quantumization for graph
    for cond in list(rawdata.keys()):
      img = np.zeros([size, size, 3])
      ref = {'c':0, 'v':1, 'm':2}
      for cvm in ['c','v','m']:
        # normalize current values to (0,1) float
        rawdata[cond][cvm] -= gmin[cvm]
        rawdata[cond][cvm] /= (gmax[cvm] - gmin[cvm])
        # scale current values to (0,aSize-1) float
        rawdata[cond][cvm] *= (size-1)
        # quantumize current valutes to integer
        rawdata[cond][cvm] = rawdata[cond][cvm].astype(int)
        # drawing image
        if human:  # increase default brightness of plots
          offset = int(len(rawdata[cond][cvm]) / size)
          for (x,y) in rawdata[cond][cvm]:
            img[int(x),int(y),ref[cvm]] = offset
        for (x,y) in rawdata[cond][cvm]:
          img[int(x),int(y),ref[cvm]] += 1
        img[:,:,ref[cvm]] /= img[:,:,ref[cvm]].max()
      img *= 255
      img = img.astype(int)
      images[cond] = np.uint8(img)

    return images


if __name__ == '__main__':
  if len(sys.argv) == 5:
    input_ = sys.argv[1]
    output = sys.argv[2]
    s = int(sys.argv[3]) - 1
    e = int(sys.argv[4])
    human = False
  else:
    print('no args: run in test mode')
    input_ = '/mnt/nas/data/waveforms/20171024/25Hz-50%/25Hz-50%-'
    output = '/home/camus/data/project/25Hz-50%-'
    s = 0
    e = 1
    human = True

  for i in range(s,e):
    startTime = time.time()
    number = str(i+1).zfill(3)
    try:
      ImageGen().run(input_+number+'.csv', output+number+'.png', human=human)
    except:
      continue
    print('processed: '+input_+number+'.csv'+'  time: %f'%(time.time()-startTime))
