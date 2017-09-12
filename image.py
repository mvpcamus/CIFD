from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im

INPUT_PHASES = 2          # number of phases in data
IMG_SIZE = (1000,1000,3)  # image size
INPUT_FILE = '/home/camus/data/waveforms/20170830/20170830-25Hz-50%-001.csv'
FAULT_FILE = '/home/camus/data/waveforms/20170830/20170830-25Hz-50%-001_fault.csv'
OUTPUT = 'F'              # 'N': normal data graph, 'F': fault data graph
FAULT_TYPE = 'S'          # output fault type. 'S':stator, 'R':rotor, 'B':bearing
FOR_HUMAN = True          # set brightness augmentation of plots for human eyes
SAVE = True               # set to save image or just to show
IMG_FILE = './output.png'

##### read in csv files #####
''' read normal data
    normal[,0]: [current 1, current 2]
    normal[,1]: [voltage 1, voltage 2]
    normal[,2]: [movement 1, movement 2] '''

normal = []
with open(INPUT_FILE) as csvfile:
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
        exit()
    normal.append([[values[1],values[2]],[values[3],values[4]],[values[5],values[6]]])
  if normal == []: pass # do nothing
  else: normal = np.array(normal)

''' read fault data
    fault[,0]: [stator fault current 1, stator fault current 2]
    fault[,1]: [rotor fault current 1, rotor fault current 2]
    fault[,2]: [bearing fault current 1, bearing fault current 2] '''

fault = []
with open(FAULT_FILE) as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  reader.next()  # skip a line of index
  for row in reader:
    values = []
    for col in row:
      try:
        values.append(float(col))
      except:
        print('ERROR: inadequate fault value: ', col)
        exit()
    fault.append([[values[3],values[4]],[values[5],values[6]],[values[7],values[8]]])
  if fault == []: pass # do nothing
  else: fault = np.array(fault)

##### depict cvm (current, voltage, movement) correlation matrix #####
# data normalization and quantumization for graph
if (OUTPUT == 'N'):
  c_data = np.array(normal[:,0,:])  # current data for red ch
elif (OUTPUT == 'F'):
  if (FAULT_TYPE == 'S'):
    c_data = np.array(fault[:,0,:])
  elif (FAULT_TYPE == 'R'):
    c_data = np.array(fault[:,1,:])
  elif (FAULT_TYPE == 'B'):
    c_data = np.array(fault[:,2,:])
  else:
    print('ERROR: invalid fault type: ', FAULT_TYPE)
    exit()
else:
  print('ERROR: invalid output type: ', OUTPUT)
  exit()
v_data = np.array(normal[:,1,:])    # voltage data for green ch
m_data = np.array(normal[:,2,:])    # movement data for blue ch

data = np.array([c_data,v_data,m_data]) # cvm data array
img = np.zeros(IMG_SIZE)

for cvm in range(IMG_SIZE[2]):

  totmin = min(data[cvm][:,0].min(), data[cvm][:,1].min())
  totmax = max(data[cvm][:,0].max(), data[cvm][:,1].max())

  for i in range(INPUT_PHASES):
    # normalize current values to (0,1) float
    data[cvm][:,i] -= totmin
    data[cvm][:,i] /= (totmax-totmin)

    # scale current values to (0,IMG_SIZE-1) float
    data[cvm][:,i] *= (IMG_SIZE[i]-1)

    # quantumize current valutes to integer
    data[cvm][:,i] = data[cvm][:,i].astype(int)

  # drawing image
  if FOR_HUMAN:  # increase default brightness of plots
    offset = int(np.shape(data[cvm])[0]) / IMG_SIZE[0]
    for (x,y) in data[cvm][:]:
      img[int(x),int(y),cvm] = offset

  for (x,y) in data[cvm][:]:
    img[int(x),int(y),cvm] += 1

  img[:,:,cvm] /= img[:,:,cvm].max()

img *= 255
img = img.astype(int)
img = np.uint8(img)

if SAVE:
  im.imsave(IMG_FILE, img)
else:
  plt.imshow(img)
  plt.show()

