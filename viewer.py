import sys
import matplotlib.pyplot as plt
import matplotlib.image as img

try:
  input_path = sys.argv[1]
except:
  input_path = '../images/20171016/25Hz-50%-001'

fig = plt.figure()
image = []

for i in range(4):
  name = input_path+'-'+str(i)+'.png'
  image.append(img.imread(name))
  subplot = fig.add_subplot(2, 2, i+1)
  subplot.imshow(image[i])

plt.show()


