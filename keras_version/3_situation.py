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
# matplotlib.use('agg')

class Adagrad2(Adagrad):
    def __init__(self, norm_val=[], clipvalue2 = 10.0, **kwargs):
        super(Adagrad2, self).__init__(**kwargs)
        self.norm_val = norm_val
        self.clipvalue2 = clipvalue2

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        # grads = [grad for x,grad in zip(params,grads)]
        # print(self.norm_val)
        # print(params)
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) if x in self.norm_val else g for x,g in zip(params,grads)]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) if x in self.norm_val else K.clip(g, -self.clipvalue2, self.clipvalue2) for x,g in zip(params,grads)]
        return grads

content_image_path = 'train_content/1/COCO_train2014_000000000034.jpg'
style_image_path = 'train_style/1/122.jpg'

# dimensions of the generated picture.
img_nrows = 400
img_ncols = 400
scale = 1/1.

style_weight = 0.75e-2
content_weight = 4e-4
tv_weight = 0.6e-8  # total variance
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
            # ('cin', ((5, 5, 3, 16), 16)),
            # ('c1', ((3, 3, 16, 16), 16)),
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
        # ('c4', ((3, 3, 48, 16), 16))]
        # ('cout', ((7, 7, 16, 3), 3))]
    parms = Parm()
    parms2 = Parm()

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
                x = InstanceNormalization2D2()(x)
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
    content_weights = [10.0, 1.0, 0.8, 1.0, 1.0]
    layer_features = map(lambda x: outputs_dict[x], content_layers)
    content_feature_model = Model(vgg.input,layer_features)
    content_feature_model.trainable = False

    style_layers = ['block3_conv2',
                    'block4_conv3','block1_conv2',
                    'block2_conv2','block5_conv2']
    style_weights = [1.0,  0.5, 1.0, 1.0, 10.0]
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


    def total_variation_loss(x):
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':
            a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
            b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
        else:
            a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
            b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        c = K.square(K.relu(0. - x[:, :, :, :]))
        d = K.square(K.relu(x[:, :, :, :] - 255.))

        return K.sum(K.batch_flatten(K.pow(a + b, 1.25)),axis = -1,keepdims = True) + K.sum(K.batch_flatten(c + d),axis = -1,keepdims = True) * 1

    base_image_content_features = [layer for layer in (content_feature_model(Lambda(lambda x: vgg_pre(x))(input)))]
    combination_content_features = [layer for layer in (content_feature_model(Lambda(lambda x: vgg_pre(x))(output)))]
    content_loss_list = []
    for i in range(len(content_layers)):
        content_loss_lambda = Lambda(
            lambda x: content_weights[i] * K.mean(K.batch_flatten(K.square(x[0] - x[1])), axis=-1,
                                                  keepdims=True))
        content_loss_list.append(content_loss_lambda(
            [base_image_content_features[i], combination_content_features[i]]))
    content_loss = add(content_loss_list)


    base_image_style_features = [InstanceNormalization2D()(layer) for layer in (style_feature_model(Lambda(lambda x:vgg_pre(x))(input2)))]
    combination_style_features = [InstanceNormalization2D()(layer) for layer in (style_feature_model(Lambda(lambda x:vgg_pre(x))(output)))]
    style_loss_list = []
    for i in range(len(style_layers)):
        style_loss_lambda = Lambda(lambda x: style_weights[i] * K.mean(
            K.batch_flatten(K.square(Mean_std()(x[0]) - Mean_std()(x[1]))), axis=-1, keepdims=True))
        style_loss_list.append(style_loss_lambda([base_image_style_features[i], combination_style_features[i]]))
    style_loss = add(style_loss_list)

    tv_loss_lambda = Lambda(lambda x:total_variation_loss(x))
    tv_loss = tv_loss_lambda(output)

    total_loss = Lambda(lambda x: x[0] * style_weight + x[1] * content_weight + x[2] * tv_weight, name='totalloss')([style_loss,content_loss,tv_loss])
    loss_model = Model(inputs=[input,input2], outputs=[content_loss,style_loss,tv_loss])
    loss_model_debug = Model(inputs=[input, input2],
                             outputs=style_loss_list + content_loss_list)
    train_model = Model(inputs=[input,input2], outputs=total_loss)

    return g_model,loss_model,train_model,meta_model,loss_model_debug



batch_size = 4
train_datagen = ImageDataGenerator(
        rescale=1.0)

content_generator = train_datagen.flow_from_directory(
        'train_content',
        target_size=(img_nrows, img_ncols),
        batch_size=batch_size * 30 * 4,
        class_mode='binary')

style_generator = train_datagen.flow_from_directory(
        'train_style',
        target_size=(img_nrows, img_ncols),
        batch_size=batch_size,
        class_mode='binary')



g_model,loss_model,train_model,meta_model,loss_model_debug = get_Net(style_weight = style_weight,content_weight = content_weight,tv_weight =tv_weight)
g_model.summary()



class LossEvaluation(Callback):
    def __init__(self, validation_img = None, interval=1, loss_model = None):
        super(Callback, self).__init__()
        self.interval = interval
        self.loss_model = loss_model
        self.validation_img = validation_img

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.loss_model.predict(self.validation_img, verbose=0, batch_size=2 ** 9)
            content_loss, style_loss, tv_loss = np.mean(y_pred, axis=1)
            print("epoch: %d - content_loss: %.3f, style_loss: %.3f, tv_loss: %.3f, total_loss: %.3f " %
                  (epoch + 1, content_loss * content_weight, style_loss * style_weight, tv_loss * tv_weight,
                   content_loss * content_weight+ style_loss * style_weight + tv_loss * tv_weight))



print('train')
count = 0
end = 1200
lr = 0.0005
val = preprocess_image_vgg19('./c.jpg')
style_val2 = preprocess_image_vgg19('./picasso.jpg')
style_val = preprocess_image_vgg19('./a.jpg')
weights_norm = meta_model.trainable_weights
opt = Adam(lr = 0.00004, clipnorm=2, clipvalue=5e0)
train_model.compile(loss=lambda y_true, y_pred: y_pred,optimizer=opt)
data_x = []
data_x2 = []
style_img = preprocess_image(style_generator.next()[0][1])
beta = 30 * 4
ep = 1
for content_images,_ in content_generator:
    count += 1
    if count % 50 == 0 and count < 60:
        print('-----')
        opt = Adagrad2(lr = 0.0001, clipnorm=2, clipvalue=5e0,norm_val = weights_norm, clipvalue2=2e0)
        train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=opt)
        beta = 15 * 4
        ep = 1
    lossEvaluation = LossEvaluation([np.array([val]), np.array([style_val]).repeat(1, axis=0)], 2, loss_model)
    lossEvaluation2 = LossEvaluation([np.array([val]), np.array([style_val2]).repeat(1, axis=0)], 2, loss_model)

    for i in range(len(content_images)):
        content_images[i] = preprocess_image(content_images[i])
        data_x.append(content_images[i])
        data_x2.append(style_img)
        if i % (batch_size*beta)== 0:
            style_img = preprocess_image(style_generator.next()[0][1])
            pass

    train_model.fit([np.array(data_x), np.array(data_x2)], np.array([0] * np.array(data_x).shape[0]),
                    batch_size=batch_size/2, epochs=ep,
                    shuffle=True, verbose=0, callbacks=[lossEvaluation,lossEvaluation2])
    data_x = []
    data_x2 = []

    if count % 3 == 0:
        print count
        output_image = g_model.predict([np.array([val]), np.array([style_val])])
        imsave('debug/output48_%s.jpg'%(count%100), deprocess_image(output_image[0].copy()))
        output_image = g_model.predict([np.array([val]), np.array([style_val2])])
        imsave('debug/output482_%s.jpg' %(count%100), deprocess_image(output_image[0].copy()))
        if count == end:
            break

train_model.save_weights('./s7.h5',overwrite=True)
g_model.save_weights('./g7.h5',overwrite=True)

