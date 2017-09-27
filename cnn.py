from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

print('Tensorflow version ' + tf.__version__)

class DataGen(object):
  '''
  Generate input data batches from png images
  args
    file_path  = input image(png) file path
    batch_size = batch size for each training step
    one_hot    = True: output labels (Y_) as one_hot arrays
    shuffle    = True: shuffle data queue sequence
  return
    data = {'X':[X1, ..., Xn], 'Y_':[Y_1, ..., Y_n], 'png':[png_path1, ..., png_pathn]}
  '''
  def __init__(self, file_path, batch_size=1, one_hot=True, shuffle=True):
    self.data = {}
    files = [os.path.join(file_path, s) for s in os.listdir(file_path)]
    queue = tf.train.string_input_producer(files, shuffle=shuffle)
    reader = tf.WholeFileReader()
    file_path, contents = reader.read(queue)
    img = tf.image.decode_png(contents, channels=3)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      self.data['png'] = [sess.run(file_path) for _ in files]
      self.data['X'] = [sess.run(img) for _ in files]
      self.data['Y_'] = [int(p.strip('.png').split('-')[3]) for p in self.data['png']]
      if one_hot: self.data['Y_'] = sess.run(tf.one_hot(self.data['Y_'],4))
      coord.request_stop()
      coord.join(threads)

  def out(self):
    return self.data


class ConvLayer(object):
  '''
  Construct a convolutional 2D layer and its summaries
  args:
    image  = input image array
    ch_in  = input image channel size (e.g. rgb = 3)
    ch_out = output channel size (number of kernels)
    size   = size of kernel (patch)
    stride = kernel (patch) stride
  '''
  def __init__(self, image, ch_in, ch_out, size, stride):
    self.img = image
    self.strd = stride
    _W_shape = [size, size, ch_in, ch_out]
    self.W = tf.Variable(tf.truncated_normal(_W_shape, stddev=0.1), trainable=True, name='W')
    self._summary(self.W, 'W')
    self.B = tf.Variable(tf.constant(0.1, tf.float32, [ch_out]), trainable=True, name='B')
    self._summary(self.B, 'B')

  def out(self):
    return tf.nn.relu(
        tf.nn.conv2d(self.img, self.W, strides=[1, self.strd, self.strd, 1], padding='SAME') + self.B)

  @staticmethod
  def _summary(var, name):
    _mean = tf.reduce_mean(var)
    _variance = tf.reduce_mean(tf.square(var - _mean))
    tf.summary.scalar(name+'_mean', _mean)
    tf.summary.scalar(name+'_variance', _variance)
    tf.summary.histogram(name, var)


class FCLayer(object):
  '''
  Construct a fully connected layer and its summaries
  args:
    input_ = input array
    n_in   = input size
    n_out  = output size
    relu   = pass through ReLU as activation function (True or False)
  '''
  def __init__(self, input_, n_in, n_out, relu=False):
    self.input_ = input_
    self.n_in = n_in
    self.n_out = n_out
    self.relu = relu
    self.W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1), trainable=True, name='W')
    self._summary(self.W, 'W')
    self.B = tf.Variable(tf.constant(0.0, tf.float32, [n_out]), trainable=True, name='B')
    self._summary(self.B, 'B')

  def out(self):
    if self.relu:
      return tf.nn.relu(tf.matmul(self.input_, self.W) + self.B)
    else:
      return tf.matmul(self.input_, self.W) + self.B

  @staticmethod
  def _summary(var, name):
    _mean = tf.reduce_mean(var)
    _variance = tf.reduce_mean(tf.square(var - _mean))
    tf.summary.scalar(name+'_mean', _mean)
    tf.summary.scalar(name+'_variance', _variance)
    tf.summary.histogram(name, var)


class BatchNorm(object):
  '''
  Construct a batch normalization for input array
  args
    input_ = input array
    n_out  = output size
    train  = True: train phase, False: test phase
  '''
  def __init__(self, input_, n_out, train):
    self.input_ = input_
    self.beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
    self.gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(input_, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    self.mean, self.var = tf.cond(train, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))

  def out(self):
    return tf.nn.batch_normalization(self.input_, self.mean, self.var, self.beta, self.gamma, 1e-3)


def model(X, Y_, train=True):
  '''
  Define a DNN inference model
  args:
    X     = input image array
    Y_    = labels of input (solutions of Y)
    train = True: train phase, False: test phase
  returns:
    Y     = predicted output array (e.g. [1, 0, 0, 0])
    cross_entropy
    accuracy
    incorrects = indices of incorrect inference

  X:      [1000 x 1000 x 3] HWC image volume
  Conv1 : [100 x 100 x K1] output volume after [100 x 100 x 3] kernel with stride 10
  Conv2 : [50 x 50 x K2] output volume after [10 x 10 x K1] kernel with stride 2
  Conv3 : [25 x 25 x K3] output volume after [5 x 5 x K2] kernel with stride 2
  Conv4 : [25 x 25 x K4] output volume after [3 x 3 x K3] kernel with stride 1
  Full1 : [F1] output nodes from [25 x 25 x K4] input nodes
  Full2 : [F2] output nodes from [F1] input nodes
  Output: [4] ouput nodes from [F2] input nodes
  '''
  K1 = 10    # first conv. kernel depth
  K2 = 20    # second conv. kernel depth
  K3 = 40    # third conv. kernel depth
  K4 = 20    # forth conv. kernel depth
  F1 = 500   # first FC layer node size
  F2 = 50    # second FC layer node size

  with tf.variable_scope('Conv1'):
    y1 = ConvLayer(X, 3, K1, 100, 10).out()
  with tf.variable_scope('BN'):
    y1 = BatchNorm(y1, K1, train).out()
  with tf.variable_scope('Conv2'):
    y2 = ConvLayer(y1, K1, K2, 10, 2).out()
  with tf.variable_scope('Conv3'):
    y3 = ConvLayer(y2, K2, K3, 5, 2).out()
  with tf.variable_scope('Conv4'):
    y4 = ConvLayer(y3, K3, K4, 3, 1).out()
  y4_rs = tf.reshape(y4, shape=[-1, 25*25*K4])

  with tf.variable_scope('Full1'):
    y5 = FCLayer(y4_rs, 25*25*K4, F1, relu=True).out()
    y5_do = tf.cond(train, lambda:tf.nn.dropout(y5, p_keep), lambda:y5)
  with tf.variable_scope('Full2'):
    y6 = FCLayer(y5_do, F1, F2, relu=True).out()
    y6_do = tf.cond(train, lambda:tf.nn.dropout(y6, p_keep), lambda:y6)
  with tf.variable_scope('Output'):
    Ylogits = FCLayer(y6_do, F2, 4).out()
  Y = tf.nn.softmax(Ylogits, name='Y')

  with tf.variable_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)
  with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    incorrects = tf.squeeze(tf.where(tf.logical_not(correct_prediction)), [1])

  return Y, cross_entropy, accuracy, incorrects


if __name__ == '__main__':
  train = True

  if train:
    TOTAL_STEP = 3000
    BATCH_SIZE = 120
    INPUT_PATH = '/mnt/data/camus/images/20170919/'
    MODEL_PATH = './tmp/train/model/cnn.ckpt'
    LOG_DIR = './tmp/train/log/'

    startTime = time.time()
    # input generation
    with tf.Graph().as_default() as input_g:
      data = DataGen(INPUT_PATH, BATCH_SIZE).out()
    print('[%6.2f] successfully generated train data'%(time.time()-startTime))

    # training phase
    with tf.Graph().as_default() as train_g:
      # input X: 1000 x 1000 rgb color image
      X = tf.placeholder(tf.float32, [None, 1000, 1000, 3], name='X')
      # target values Y_: 0=normal, 1=stator fault, 2=rotor fault, 3=bearing fault
      Y_ = tf.placeholder(tf.float32, [None, 4], name='Y_')
      with tf.variable_scope('Config'):
        # dropout keep probability
        p_keep = tf.placeholder(tf.float32, name='p_keep')
        train = tf.placeholder(tf.bool, name='train')
        # learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.001, global_step, int(TOTAL_STEP/5), 0.5, staircase=True, name='lr')

      # load inference model
      Y, cross_entropy, accuracy, _ = model(X, Y_, train)
      train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)

      with tf.variable_scope('Metrics'):
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('learning_rate', lr)

      # set tensorboard summary and saver
      merged = tf.summary.merge_all()
      saver = tf.train.Saver(max_to_keep=None)

      # training session
      print('----- training start -----')
      with tf.Session() as sess:
        sum_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        tf.global_variables_initializer().run()
        step = 1
        length = len(data['png'])
        while step <= TOTAL_STEP:
          for batch in range(int(length/BATCH_SIZE)+1):
            s = batch*BATCH_SIZE
            e = (batch+1)*BATCH_SIZE if (batch+1)*BATCH_SIZE < length else length
            if e <= s: break
            _, summary, acc, ent = sess.run([train_op, merged, accuracy, cross_entropy],
                                            {X:data['X'][s:e], Y_:data['Y_'][s:e], p_keep:0.75, train:True})
            sum_writer.add_summary(summary, step)
            print('[%6.2f] step:%3d, size:%3d, lr:%f, accuracy:%f, cross entropy:%f'
                  %(time.time()-startTime, step, e-s, lr.eval(), acc, ent))
            saver.save(sess, MODEL_PATH, global_step=step)
            step += 1
            if step > TOTAL_STEP: break
      print('-----  training end  -----')

  else:
    BATCH_SIZE = 100
    INPUT_PATH = '/mnt/data/camus/images/20170830/'
    MODEL_PATH = './tmp/test/model/1k/cnn.ckpt'
    LOG_DIR = './tmp/test/log/'

    startTime = time.time()
    # input generation
    with tf.Graph().as_default() as input_g:
      data = DataGen(INPUT_PATH, shuffle=False).out()
    print('[%6.2f] successfully generated test data'%(time.time()-startTime))

    # test phase
    with tf.Graph().as_default() as test_g:
      # input X: 1000 x 1000 rgb color image
      X = tf.placeholder(tf.float32, [None, 1000, 1000, 3], name='X')
      # target values Y_: 0=normal, 1=stator fault, 2=rotor fault, 3=bearing fault
      Y_ = tf.placeholder(tf.float32, [None, 4], name='Y_')
      with tf.variable_scope('Config'):
        # dropout keep probability
        p_keep = tf.placeholder(tf.float32, name='p_keep')
        train = tf.placeholder(tf.bool, name='train')

      # load inference model
      Y, cross_entropy, accuracy, incorrects = model(X, Y_, train)

      with tf.variable_scope('Metrics'):
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('cross_entropy', cross_entropy)

      # set tensorboard summary and saver
      merged = tf.summary.merge_all()
      saver = tf.train.Saver()

      # test session
      print('----- test start -----')
      with tf.Session() as sess:
        sum_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        tf.global_variables_initializer().run()
        saver.restore(sess, MODEL_PATH)
        avg_accuracy = 0
        length = len(data['png'])
        for step in range(int(length/BATCH_SIZE)+1):
          s = step*BATCH_SIZE
          e = (step+1)*BATCH_SIZE if (step+1)*BATCH_SIZE < length else length
          if e <= s: break
          summary, acc, ent, incor, y_, y = sess.run([merged, accuracy, cross_entropy, incorrects, Y_, Y],
                                                {X:data['X'][s:e], Y_:data['Y_'][s:e], p_keep:1.00, train:False})
          sum_writer.add_summary(summary, step+1)
          avg_accuracy += acc * (e-s)
          print('[%6.2f] steps:%d, size:%d, accuracy:%f, cross entropy:%f'
                  %(time.time()-startTime, step+1, e-s, acc, ent))
          if len(incor) > 0: print('   incorrects list:')
          for i in incor:
            print('   [%3d] Answer:Infer = %d:%d  at %s'
                    %(s+i, tf.argmax(y_[i],0).eval(),tf.argmax(y[i],0).eval(),data['png'][s+i]))
        print('-----  test end  -----')
        print('[%6.2f] total average accuracy: %f'%(time.time()-startTime, avg_accuracy/length))
