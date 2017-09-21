from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

print('Tensorflow version ' + tf.__version__)

TOTAL_STEP = 400
INPUT_DIR = '/mnt/data/camus/images/20170919/'
MODEL_DIR = './tmp/model/'
LOG_DIR = './tmp/log/'

# input X: 1000 x 1000 rgb color image
X = tf.placeholder(tf.float32, [None, 1000, 1000, 3])
# target values Y_: 0=normal, 1=stator fault, 2=rotor fault, 3=bearing fault
Y_ = tf.placeholder(tf.float32, [None, 4])
# learning rate
global_step = tf.Variable(0, name='global_step', trainable=False) 
lr = tf.train.exponential_decay(0.001, global_step, int(TOTAL_STEP/5), 0.5, staircase=True)
#lr = tf.placeholder(tf.float32)
# dropout keep probability
p_keep = tf.placeholder(tf.float32)

phase_train = tf.placeholder(tf.bool)

def batch_norm(x, n_out, phase_train):
  with tf.variable_scope('bn'):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

# devide list l into (length n) lists
def chunks(l, n):
  for i in range(0, len(l[0]), n):
    yield l[0][i:i+n], l[1][i:i+n]

def summaries(var, name):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)

K = 25    # first conv. output depth
L = 50    # send conv. output depth
M = 25    # third conv.
N = 100   # fully connected layer

W1 = tf.Variable(tf.truncated_normal([100, 100, 3, K], stddev=0.1), name='W1')
summaries(W1, 'W1')
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name='B1')
summaries(B1, 'B1')
W2 = tf.Variable(tf.truncated_normal([50, 50, K, L], stddev=0.1), name='W2')
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]), name='B2')
W3 = tf.Variable(tf.truncated_normal([25, 25, L, M], stddev=0.1), name='W3')
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]), name='B3')

W4 = tf.Variable(tf.truncated_normal([25 * 25 * M, N], stddev=0.1), name='W4')
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]), name='B4')
W5 = tf.Variable(tf.truncated_normal([N, 4], stddev=0.1), name='W5')
B5 = tf.Variable(tf.constant(0.1, tf.float32, [4]), name='B5')

stride = 10
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
Y1 = batch_norm(Y1, K, phase_train)
stride = 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
#Y2 = batch_norm(Y2, L, phase_train)
stride = 2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
#Y3 = batch_norm(Y3, M, phase_train)
YY = tf.reshape(Y3, shape=[-1, 25 * 25 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, p_keep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits, name='Y')

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)

# read input png
files = os.listdir(INPUT_DIR)
files = [os.path.join(INPUT_DIR,s) for s in files]

file_queue = tf.train.string_input_producer(files, shuffle=False)
image_reader = tf.WholeFileReader()

img_path, img_bin = image_reader.read(file_queue)
image = tf.image.decode_png(img_bin, channels=3)

# set tensorboard summary and saver
merged = tf.summary.merge_all()
saver = tf.train.Saver()

# start session
startTime = time.time()
with tf.Session() as sess:
  sum_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

  tf.global_variables_initializer().run()

  print('----- training start -----')
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  X_batch = []
  Y_batch = []
  for i in range(len(files)):
    # image shape = 1000 x 1000 x 3 = [1000, 1000, 3]
    X_i, Y_i = sess.run([image, img_path])
    X_batch.append(X_i)
    Y_batch.append(int(Y_i.strip('.png').split('-')[3]))

  coord.request_stop()
  coord.join(threads)

  xys = []
  for x, y in chunks((X_batch,Y_batch), 50):
    xys.append((x,y))
 
  print('[%6.2f] input generated'%(time.time()-startTime))

  for step in range(int(TOTAL_STEP/len(xys))):
    i = 0
    for x_chunk, y_chunk in xys:
      y_chunk = tf.one_hot(y_chunk,4).eval()
      _, summary, acc, ent, y_predict = sess.run([train_step, merged, accuracy, cross_entropy, Y],
                                     {X:x_chunk, Y_:y_chunk, p_keep:0.75, phase_train:True})
      sum_writer.add_summary(summary, step*len(xys)+i+1)
      print('[%6.2f] step:%3d, lr:%f, accuracy:%f, entropy:%f'
            %(time.time()-startTime, step*len(xys)+i+1, lr.eval(), acc, ent))
#      print(y_predict) #PROTO
      i += 1
    saver.save(sess, MODEL_DIR+'cnn.ckpt', global_step = step)
  print('----- training end -----')
