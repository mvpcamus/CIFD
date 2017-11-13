# Copyright kairos03. All Right Reserved.

import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class StyledLabel(QLabel):
  def __init__(self, text, id=None, font_size='20pt', color='black', background=None, highlight=None, border=None):
    super().__init__()
    self.text = text
    self.id = text if id is None else id
    self.font_size = font_size
    self.color = color
    self.background = background
    self.highlight = highlight
    self.border = border

    self.init_ui()

  def init_ui(self):
    self.setText(self.text)
    self.setMargin(10)
    self.setStyleSheet(self.styleSheet() + 'font-size: %s;' % (self.font_size))
    self.setStyleSheet(self.styleSheet() + 'color: %s;' % (self.color))
    self.setStyleSheet(self.styleSheet() + 'background-color: %s;' % (self.background))
    self.setStyleSheet(self.styleSheet() + 'border: %s;' % (self.border))


class TopLabel(StyledLabel):
  def __init__(self, text, id):
    super().__init__(text, id=id, highlight='skyblue', border='1px solid black')

  @pyqtSlot(int)
  def highlighting(self, state):
    if state == self.id:
      self.setStyleSheet(self.styleSheet() + 'background-color: %s;' % (self.highlight))
    else:
      self.setStyleSheet(self.styleSheet() + 'background-color: None;')
#   print('dlog/change background, id: %s, text: %s, color: %s' % (self.id, self.text, self.height))

  def blink(self):
    for i in range(5):
      self.set_background(self.highlight)
      time.sleep(0.5)
      self.set_background(self.background)
      time.sleep(0.5)

  def make_connection(self, data_loader):
    data_loader.connect(self.highlighting)


class ResultLabel(StyledLabel):
  def __init__(self, text, id):
    super().__init__(text, id=id, border='1px solid black')

  @pyqtSlot(int, float)
  def change_background(self, state, value):
    if state == self.id:
      if state == 0:
        self.background = 'green'
      elif value >= 1.0:
        self.background = 'red'
      elif value >= 0.85:
        self.background = 'orange'
      else:
        self.background = 'yellow'
      self.setStyleSheet(self.styleSheet() + 'background-color: %s;' % (self.background))
    else:
      self.setStyleSheet(self.styleSheet() + 'background-color: None;')
#   print('dlog/change background, id: %s, text: %s, color: %s' % (self.id, self.text, self.background))

  def blink(self):
    for i in range(5):
      self.set_background(self.highlight)
      time.sleep(0.5)
      self.set_background(self.background)
      time.sleep(0.5)

  def make_connection(self, data_loader):
    data_loader.connect(self.change_background)


class ImgLabel(QLabel):
  def __init__(self, id=None, img_path=None, post_fix='', img_width=None, img_height=None):
    super().__init__()
    self.id = id
    self.default_path = './demo/white.png'
    self.post_fix = post_fix
    self.img_path = img_path + post_fix if img_path is not None else self.default_path
    self.img_width = img_width
    self.img_height = img_height

    self.init_ui()

  def init_ui(self):
    pixmap = self.make_pixmap()
    self.setPixmap(pixmap)
    self.setMaximumSize(self.img_width, self.img_height)

  def make_pixmap(self):
    pixmap = QPixmap(self.img_path)
    if self.img_width is not None and self.img_height is not None:
      pixmap = pixmap.scaled(self.img_width, self.img_height, Qt.KeepAspectRatio)
    else:
      self.img_width = pixmap.width()
      self.img_height = pixmap.height()
    return pixmap

  @pyqtSlot(int, str)
  def reload_image(self, layer, img_path):
    if layer == self.id or layer == 0:
      if img_path != self.default_path:
        img_path = img_path + self.post_fix

      self.img_path = img_path
      pixmap = self.make_pixmap()
      self.setPixmap(pixmap)
#     print("dlog/reload image, layer: %s, path: %s" % (layer, img_path))

  def make_connection(self, data_loader):
    data_loader.connect(self.reload_image)


class DataLoader(QObject):
  def __init__(self):
    super().__init__()

  setState = pyqtSignal(int)
  inputImagePathLoaded = pyqtSignal(int, str)
  convImagePathLoaded = pyqtSignal(int, str)
  outputStateValueLoaded = pyqtSignal(int, float)

  def on_reset(self):
    self.setState.emit(0)
    self.inputImagePathLoaded.emit(0, './demo/white.png')
    self.convImagePathLoaded.emit(0, './demo/white.png')
    self.outputStateValueLoaded.emit(-1, 0)
    self.setState.emit(1)

  def on_input_data_loaded(self, path):
    self.inputImagePathLoaded.emit(0, path)

  def on_conv_data_loaded(self, path):
    self.setState.emit(2)
    for i in range(1, 5):
      self.convImagePathLoaded.emit(i, path)
      time.sleep(0.3)

  def on_output_data_loaded(self, state_code, value):
    self.setState.emit(3)
    self.outputStateValueLoaded.emit(state_code, value)
