#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# References
    - (http://arxiv.org/abs/1709.04111)
'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, DirectoryIterator, ImageDataGenerator
from keras.applications import vgg19
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from util import *
import matplotlib
from scipy.misc import imsave
from joblib import dump, load
import sys


# dimensions of the generated picture.
img_nrows = 400
img_ncols = 400
scale = 1/1.

style_weight = 0.75e-2
content_weight = 3e-4
tv_weight = 0.8e-8  # total variance
print style_weight,content_weight,tv_weight,scale

def preprocess_image_vgg19(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img[:, :, :, ::-1] * scale
    img = np.squeeze(img, axis=0)
    return img

def preprocess_image(img):
    img = img[ :, :, ::-1] * scale
    return img*1.0

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    x = x[:, :, ::-1] / scale
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def imshow(image, title=None):
    image = np.array(image).astype('uint8')
    plt.imshow(image)
    if title is not None:
        plt.title(title)

channels = 3



def get_Net(input_shape = (img_nrows,img_ncols,channels),style_weight = style_weight,content_weight = content_weight,tv_weight =tv_weight):
    import collections
    residual_size = 48

    class Parm(object):
        params_pos = 0
        params_pos2 = 0
        params_dict = [
            ('c2', ((3, 3, 32, residual_size), residual_size)),
            ('rc1-1', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc1-2', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc2-1', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc2-2', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc3-1', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc3-2', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc4-1', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc4-2', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc5-1', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc5-2', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc6-1', ((3, 3, residual_size, residual_size), residual_size)),
            ('rc6-2', ((3, 3, residual_size, residual_size), residual_size)),
            ('c3', ((3, 3, residual_size, 48), 48))]
    parms = Parm()

    class ConvLayer(object):
        def __init__(self, filters, kernel_size=3, stride=1,
                     upsample=None, instance_norm=False, activation='relu', trainable = False, parms = None):
            super(ConvLayer, self).__init__()
            self.upsample = upsample
            self.filters = filters
            self.kernel_size = kernel_size
            self.stride = stride
            self.activation = activation
            self.instance_norm = instance_norm
            self.trainable = trainable
            self.parms = parms

        def __call__(self, inputs):
            if self.trainable:
                x = inputs
            else:
                x = inputs[0]
            name = None
            if self.parms:
                params_pos = self.parms.params_pos
                params_pos2 = self.parms.params_pos2
                params_dict = self.parms.params_dict
                weight_start = params_pos2
                weight_shape = params_dict[params_pos][1]
                name = params_dict[params_pos][0]
                self.parms.params_pos += 1
                self.parms.params_pos2 += np.prod(weight_shape[0]) + weight_shape[1]
            if self.upsample:
                x = UpSampling2D(size=(self.upsample, self.upsample))(x)
            x = ReflectPadding2D(self.kernel_size//2)(x)
            if self.activation == 'prelu':
                if self.trainable:
                    x = Conv2D(self.filters, self.kernel_size, strides=(self.stride, self.stride),name = name)(x)
                else:
                    x = MyConv2D(self.filters, self.kernel_size, strides=(self.stride, self.stride),shape = (weight_start,weight_shape),name = name)([x,inputs[1]])
                x = PReLU()(x)
            else:
                if self.trainable:
                    x = Conv2D(self.filters, self.kernel_size, strides=(self.stride, self.stride),
                               activation=self.activation,name = name)(x)
                else:
                    x = MyConv2D(self.filters, self.kernel_size, strides=(self.stride, self.stride),
                                 activation=self.activation,shape = (weight_start,weight_shape),name = name)([x,inputs[1]])
            if self.instance_norm:
                x = InstanceNormalization2D()(x)
            return x

    class ResidualBlock(object):
        def __call__(self, inputs):
            if self.trainable:
                x = inputs
                out = self.conv1(x)
                out = self.conv2(out)
            else:
                x = inputs[0]
                out = self.conv1([x, inputs[1]])
                out = self.conv2([out, inputs[1]])
            out = add([out, x])
            return out

        def __init__(self, filters, parms, trainable = False):
            super(ResidualBlock, self).__init__()
            self.conv1 = ConvLayer(filters, kernel_size=3, stride=1, parms = parms, trainable = trainable)
            self.conv2 = ConvLayer(filters, kernel_size=3, stride=1, parms = parms, trainable = trainable, activation=None)
            self.parms = parms
            self.trainable = trainable

    def vgg_pre(x):
        mul = np.array([1.0]*3).reshape(1,1,1,3)
        meanoffset = (np.array([-103.939,-116.779,-123.68]) * scale).reshape(1,1,1,3)
        assert K.ndim(x) == 4
        return (x + meanoffset)/mul

    def gram_matrix(features,shape):
        batch, ch, h, w = shape
        if K.image_data_format() != 'channels_first':
            features = K.permute_dimensions(features, (0, 3, 1, 2))
            batch, h, w,ch = shape
        features = K.reshape(features,(-1, ch, h * w))
        gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1))) / ((int(ch) * int(h) * int(w) * 10))
        return gram

    input = Input(shape=(input_shape), dtype="float32")  # content
    input2 = Input(shape=(input_shape), dtype="float32")  # style

    # vgg
    vgg = vgg19.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    content_layers = ['block5_conv2',
                      'block4_conv3','block1_conv2']
    layer_features = map(lambda x: outputs_dict[x], content_layers)
    content_feature_model = Model(vgg.input,layer_features)
    content_feature_model.trainable = False

    style_layers = ['block3_conv2',
                    'block4_conv3','block1_conv2',
                    'block2_conv2','block5_conv2']
    layer_features = map(lambda x: outputs_dict[x], style_layers)
    style_feature_model = Model(vgg.input, layer_features)
    style_feature_model.trainable = False


    # meta
    meta_input1 = concatenate([Mean_std()(l) for l in style_feature_model(Lambda(lambda x: vgg_pre(x))(input2))])
    x = Lambda(lambda x: 1/15 * x)(input2)
    x = ConvLayer(64, kernel_size=9, stride=1, trainable=True)(x)
    x = ConvLayer(64, kernel_size=3, stride=2, trainable=True)(x)
    x = ConvLayer(256, kernel_size=3, stride=2, trainable=True)(x)
    meta_input2 = Mean_std()(x)
    meta_output_list = []
    hidden = concatenate([Dense(128, activation='relu')(meta_input1), Dense(16, activation='sigmoid')(meta_input2),
                          Dense(128, activation='relu')(meta_input2)])
    hidden = Dropout(0.05)(hidden)
    for name,shape in parms.params_dict:
        hidden2 = concatenate([Dense(64,activation = 'relu')(meta_input1),Dense(16,activation = 'sigmoid')(meta_input2),Dense(6,activation = 'relu')(meta_input2)])
        hidden2 = concatenate([Dropout(0.05)(hidden2), hidden])
        kernel = Dense(np.prod(shape[0]),activation = 'linear')(hidden2)
        bias = Dense(shape[1],activation = 'linear')(hidden2)
        meta_output_list.append(Lambda(lambda x:0.0001*x)(kernel))
        meta_output_list.append(Lambda(lambda x:0.0001*x)(bias))
    meta_output = (concatenate(meta_output_list))

    meta_model = Model(input2,meta_output)

    # transform
    in_encode = ConvLayer(16, kernel_size=9, stride=1,  trainable=True)
    x = Lambda(lambda x: 1/15. * x)(input)
    x = in_encode(x)
    x =  ConvLayer(32, kernel_size=3, stride=2, trainable=True)(x)
    x =  ConvLayer(residual_size, kernel_size=3, stride=2, parms = parms)([x,meta_output])
    for i in range(6):
        x = ResidualBlock(residual_size, parms = parms)([x,meta_output])
    x = ConvLayer(48, kernel_size=3, stride=1, upsample=2, parms = parms)([x,meta_output])
    x = ConvLayer(16, kernel_size=3, stride=1,upsample=2, trainable=True)(x)
    out_decode = ConvLayer(3, kernel_size=9, stride=1, activation=None,  trainable=True)
    x = out_decode(x)
    output = x
    output = Lambda(lambda x: x * 15.)(output)
    g_model = Model(inputs=[input,input2], outputs=output)


    return g_model


g_model = get_Net(style_weight = style_weight,content_weight = content_weight,tv_weight =tv_weight)
g_model.summary()
# g_model.load_weights('./g5_48.h5')
g_model.load_weights('./g7.h5')

val = preprocess_image_vgg19(sys.argv[1])
style_val = preprocess_image_vgg19(sys.argv[2])
opt = Adam(lr = 0.00004, clipnorm=2, clipvalue=5e0)
g_model.compile(loss='mse',optimizer=opt)
output_image = g_model.predict([np.array([val]), np.array([style_val])])
imsave(sys.argv[3], deprocess_image(output_image[0].copy()))
