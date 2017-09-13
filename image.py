from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im

def img_gen(aInput, aOutput=None, aSize=(1000,1000,3), aHuman=False):
# INPUT_PHASES = 2          # number of phases in data
# IMG_SIZE = (1000,1000,3)  # image size
# INPUT_FILE = '/home/camus/data/waveforms/20170830/20170830-25Hz-50%-001.csv'
# FAULT_FILE = '/home/camus/data/waveforms/20170830/20170830-25Hz-50%-001_fault.csv'
# OUTPUT = 'F'              # 'N': normal data graph, 'F': fault data graph
# FAULT_TYPE = 'S'          # output fault type. 'S':stator, 'R':rotor, 'B':bearing
# FOR_HUMAN = True          # set brightness augmentation of plots for human eyes
# SAVE = True               # set to save image or just to show
# IMG_FILE = './output.png'
  normal, fault = read_csv(aInput)
  images = graph(normal, fault, aSize, True)
  depict(images, aOutput)
  return

def read_csv(aInput):
  ''' read normal data
      normal[,0]: [current 1, current 2]
      normal[,1]: [voltage 1, voltage 2]
      normal[,2]: [movement 1, movement 2] '''
  normal = []
  with open(aInput+'.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    reader.next()
    reader.next()
    reader.next()  # skip three lines of index
    for row in reader:
      values = []
      for col in row:
        try:
          values.append(float(col))
        except:
          print('ERROR: inadequate data value: ', col)
          return False
      normal.append([[values[1],values[2]],[values[3],values[4]],[values[5],values[6]]])
    if normal == []: return None
    else: normal = np.array(normal)
  ''' read fault data
      fault[,0]: [stator fault current 1, stator fault current 2]
      fault[,1]: [rotor fault current 1, rotor fault current 2]
      fault[,2]: [bearing fault current 1, bearing fault current 2] '''
  fault = []
  with open(aInput+'_fault.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    reader.next()  # skip a line of index
    for row in reader:
      values = []
      for col in row:
        try:
          values.append(float(col))
        except:
          print('ERROR: inadequate fault value: ', col)
          return False
      fault.append([[values[3],values[4]],[values[5],values[6]],[values[7],values[8]]])
    if fault == []: return None
    else: fault = np.array(fault)
  return normal,fault


def graph(aNormal, aFault, aSize, aHuman):
  INPUT_PHASES = 2  # number of phases in data
  ##### depict cvm (current, voltage, movement) correlation matrix #####
  # data normalization and quantumization for graph
  c_data = [aNormal[:,0,:], aFault[:,0,:],
             aFault[:,1,:], aFault[:,2,:]] # current data  for red ch
  v_data = aNormal[:,1,:]                  # voltage data  for green ch
  m_data = aNormal[:,2,:]                  # movement data for blue ch
  images = []
  minmax = []
  # calculate min and max of each cvm for graph scaling
  for cvm in range(aSize[2]):
    minmax.append( {"min":min(aNormal[:,cvm,0].min(), aNormal[:,cvm,1].min()),
                    "max":max(aNormal[:,cvm,0].max(), aNormal[:,cvm,1].max())} )
  for cur in c_data:
    data = np.array([cur,v_data,m_data])   # cvm data array
    img = np.zeros(aSize)
    for cvm in range(aSize[2]):
      totmin = min(data[cvm][:,0].min(), data[cvm][:,1].min())
      totmax = max(data[cvm][:,1].max(), data[cvm][:,1].max())
      for i in range(INPUT_PHASES):
        # normalize current values to (0,1) float
        data[cvm][:,i] -= totmin # minmax[cvm]["min"]
        data[cvm][:,i] /= (totmax-totmin) #(minmax[cvm]["max"]-minmax[cvm]["min"])
        # scale current values to (0,aSize-1) float
        data[cvm][:,i] *= (aSize[i]-1)
        # quantumize current valutes to integer
        data[cvm][:,i] = data[cvm][:,i].astype(int)
      # drawing image
      if aHuman:  # increase default brightness of plots
        offset = int(np.shape(data[cvm])[0]) / aSize[0]
        for (x,y) in data[cvm][:]:
          img[int(x),int(y),cvm] = offset
      for (x,y) in data[cvm][:]:
        img[int(x),int(y),cvm] += 1
      img[:,:,cvm] /= img[:,:,cvm].max()
    img *= 255
    img = img.astype(int)
    images.append(np.uint8(img))
  return images


def depict(aImages, aOutput):
  titles = ['Normal','Stator Fault', 'Rotor Fault', 'Bearing Fault']
  for i in range(len(aImages)):
    if aOutput:
      im.imsave('%s-%d.png'%(aOutput,i), aImages[i])
    else:
      plt.subplot(1,len(aImages),i+1)
      plt.imshow(aImages[i])
      plt.title(titles[i])
  if not aOutput: plt.show()


if __name__ == '__main__':
  input_file = '/home/camus/data/waveforms/20170830/20170830-25Hz-50%-001'
  output_file = None #'./output'
  img_gen(input_file, output_file)
