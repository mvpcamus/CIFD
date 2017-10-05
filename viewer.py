import argparse
import matplotlib.pyplot as plt
import matplotlib.image as img

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default='../images/20170919/25Hz-50%-003')
FLAGS = parser.parse_args()

fig = plt.figure()
image = []

for i in range(4):
  name = FLAGS.i+'-'+str(i)+'.png'
  image.append(img.imread(name))
  subplot = fig.add_subplot(2, 2, i+1)
  subplot.imshow(image[i])

plt.show()


