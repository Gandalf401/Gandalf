"""
    crop_and_resize层
"""
import tensorflow as tf


class CropAndResize(tf.keras.layers.Layer):
    def __init__(self, size, left, top, height, width, mode):
        super(CropAndResize, self).__init__()
        self.size = size
        self.mode = mode
        self.box = [top, left, top + height, left + width]

    def build(self, input_shape):
        super(CropAndResize, self).build(input_shape)

    def call(self, x):
        img_h, img_w = x.shape[1], x.shape[2]
        boxes = [[self.box[0] / img_h, self.box[1] / img_w, self.box[2] / img_h, self.box[3] / img_w]]
        return tf.image.crop_and_resize(x, boxes, [0], self.size, self.mode)
