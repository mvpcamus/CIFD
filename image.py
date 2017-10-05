from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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
    import matplotlib.pyplot as plt
    import matplotlib.image as im

    rawdata = self._readCsv(csv)
    images = self._drawGraph(rawdata, size, human)

    for i in range(len(self.index)):
      if png:
        png = png.replace('.png','')
        im.imsave('%s-%d.png'%(png,i), images[self.index[i]])
      else:
        plt.subplot(1,len(self.index),i+1)
        plt.imshow(images[self.index[i]])
        plt.title(self.index[i])
    if not png: plt.show()

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
    import csv
    import copy
    import six

    rawdata = {self.index[0] : {'c':[], 'v':[], 'm':[]},
               self.index[1] : {'c':[], 'v':[], 'm':[]},
               self.index[2] : {'c':[], 'v':[], 'm':[]},
               self.index[3] : {'c':[], 'v':[], 'm':[]}}

    # read in normal data csv file
    with open(csvpath) as csvfile:
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

    # read in fault data csv file
    with open(csvpath.replace('.csv','_fault.csv')) as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      six.next(reader)  # skip a line of index in csv file
      for row in reader:
        values = []
        for col in row:
          try:
            values.append(float(col))
          except:
            print('ERROR: inadequate fault value: ', col)
            print('       in file: ', csvpath.replace('.csv','_fault.csv'))
            exit()
        rawdata[self.index[1]]['c'].append([ values[3], values[4] ])
        rawdata[self.index[2]]['c'].append([ values[5], values[6] ])
        rawdata[self.index[3]]['c'].append([ values[7], values[8] ])
      if rawdata[self.index[1]]['c'] == []:
        print('ERROR: cannot find data in file: ', csvpath.replace('.csv','_fault.csv'))
        exit()
      else:
        for cond in list(rawdata.keys()):
          if cond == self.index[0]:
            pass
          else:
            rawdata[cond]['c'] = np.array(rawdata[cond]['c'])
            rawdata[cond]['v'] = copy.deepcopy(rawdata[self.index[0]]['v'])
            rawdata[cond]['m'] = copy.deepcopy(rawdata[self.index[0]]['m'])

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
  input_file = '/home/camus/data/waveforms/20170919/25Hz-50%-003.csv'
  output_file = '/home/camus/data/samples'
  ImageGen().run(input_file, output_file, human=True)
