
# coding: utf-8

# # 导入必要的库

# In[4]:


import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import shutil
from glob import glob

from tensorboardX import SummaryWriter

import numpy as np
import multiprocessing

import copy
from tqdm import tqdm
from collections import defaultdict

import horovod.torch as hvd
import torch.utils.data.distributed

from utils import *
from models import *
import time

from pprint import pprint
display = pprint

hvd.init()
torch.cuda.set_device(hvd.local_rank())

device = torch.device("cuda:%s" %hvd.local_rank() if torch.cuda.is_available() else "cpu")


# In[5]:


is_hvd = False
tag = 'nohvd'
base = 32
style_weight = 50
content_weight = 1
tv_weight = 1e-6
epochs = 22

batch_size = 8
width = 256

verbose_hist_batch = 100
verbose_image_batch = 800

model_name = f'metanet_base{base}_style{style_weight}_tv{tv_weight}_tag{tag}'
print(f'model_name: {model_name}, rank: {hvd.rank()}')


# In[ ]:


def rmrf(path):
    try:
        shutil.rmtree(path)
    except:
        pass

for f in glob('runs/*/.AppleDouble'):
    rmrf(f)

rmrf('runs/' + model_name)


# # 搭建模型

# In[3]:


vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()


# In[4]:


transform_net = TransformNet(base).to(device)
transform_net.get_param_dict()


# In[7]:


metanet = MetaNet(transform_net.get_param_dict()).to(device)


# # 载入数据集
# 
# > During training, each content image or style image is resized to keep the smallest dimension in the range [256, 480], and randomly cropped regions of size 256 × 256.
# 
# ## 载入 COCO 数据集和 WikiArt 数据集
# 
# > The batch size of content images is 8 and the meta network is trained for 20 iterations before changing the style image.

# In[6]:


data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256/480, 1), ratio=(1, 1)), 
    transforms.ToTensor(), 
    tensor_normalizer
])

style_dataset = torchvision.datasets.ImageFolder('/home/ypw/WikiArt/', transform=data_transform)
content_dataset = torchvision.datasets.ImageFolder('/home/ypw/COCO/', transform=data_transform)

if is_hvd:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        content_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, 
        num_workers=multiprocessing.cpu_count(),sampler=train_sampler)
else:
    content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=multiprocessing.cpu_count())

if not is_hvd or hvd.rank() == 0:
    print(style_dataset)
    print('-'*20)
    print(content_dataset)


# # 测试 infer

# In[8]:


metanet.eval()
transform_net.eval()

rands = torch.rand(4, 3, 256, 256).to(device)
features = vgg16(rands);
weights = metanet(mean_std(features));
transform_net.set_weights(weights)
transformed_images = transform_net(torch.rand(4, 3, 256, 256).to(device));

if not is_hvd or hvd.rank() == 0:
    print('features:')
    display([x.shape for x in features])
    
    print('weights:')
    display([x.shape for x in weights.values()])

    print('transformed_images:')
    display(transformed_images.shape)


# # 初始化一些变量

# In[ ]:


visualization_style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
visualization_content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)


# In[ ]:


if not is_hvd or hvd.rank() == 0:
    for f in glob('runs/*/.AppleDouble'):
        rmrf(f)

    rmrf('runs/' + model_name)
    writer = SummaryWriter('runs/'+model_name)
else:
    writer = SummaryWriter('/tmp/'+model_name)

visualization_style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
visualization_content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)

writer.add_image('content_image', recover_tensor(visualization_content_images), 0)
writer.add_graph(transform_net, (rands, ))

del rands, features, weights, transformed_images


# In[ ]:


trainable_params = {}
trainable_param_shapes = {}
for model in [vgg16, transform_net, metanet]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param
            trainable_param_shapes[name] = param.shape


# # 开始训练

# In[ ]:


optimizer = optim.Adam(trainable_params.values(), 1e-3)

if is_hvd:
    optimizer = hvd.DistributedOptimizer(optimizer, 
                                         named_parameters=trainable_params.items())
    params = transform_net.state_dict()
    params.update(metanet.state_dict())
    hvd.broadcast_parameters(params, root_rank=0)


# In[ ]:


n_batch = len(content_data_loader)
metanet.train()
transform_net.train()

for epoch in range(epochs):
    smoother = defaultdict(Smooth)
    with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            n_iter = epoch*n_batch + batch
            
            # 每 20 个 batch 随机挑选一张新的风格图像，计算其特征
            if batch % 20 == 0:
                style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
                style_features = vgg16(style_image)
                style_mean_std = mean_std(style_features)
            
            # 检查纯色
            x = content_images.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue
            
            optimizer.zero_grad()
            
            # 使用风格图像生成风格模型
            weights = metanet(mean_std(style_features))
            transform_net.set_weights(weights, 0)
            
            # 使用风格模型预测风格迁移图像
            content_images = content_images.to(device)
            transformed_images = transform_net(content_images)

            # 使用 vgg16 计算特征
            content_features = vgg16(content_images)
            transformed_features = vgg16(transformed_images)
            transformed_mean_std = mean_std(transformed_features)
            
            # content loss
            content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])
            
            # style loss
            style_loss = style_weight * F.mse_loss(transformed_mean_std, 
                                                   style_mean_std.expand_as(transformed_mean_std))
            
            # total variation loss
            y = transformed_images
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
                                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
            
            # 求和
            loss = content_loss + style_loss + tv_loss 
            
            loss.backward()
            optimizer.step()
            
            smoother['content_loss'] += content_loss.item()
            smoother['style_loss'] += style_loss.item()
            smoother['tv_loss'] += tv_loss.item()
            smoother['loss'] += loss.item()
            
            max_value = max([x.max().item() for x in weights.values()])
        
            writer.add_scalar('loss/loss', loss, n_iter)
            writer.add_scalar('loss/content_loss', content_loss, n_iter)
            writer.add_scalar('loss/style_loss', style_loss, n_iter)
            writer.add_scalar('loss/total_variation', tv_loss, n_iter)
            writer.add_scalar('loss/max', max_value, n_iter)
            
            s = 'Epoch: {} '.format(epoch+1)
            s += 'Content: {:.2f} '.format(smoother['content_loss'])
            s += 'Style: {:.1f} '.format(smoother['style_loss'])
            s += 'Loss: {:.2f} '.format(smoother['loss'])
            s += 'Max: {:.2f}'.format(max_value)
            
            if (batch + 1) % verbose_image_batch == 0:
                transform_net.eval()
                visualization_transformed_images = transform_net(visualization_content_images)
                transform_net.train()
                visualization_transformed_images = torch.cat([style_image, visualization_transformed_images])
                writer.add_image('debug', recover_tensor(visualization_transformed_images), n_iter)
                del visualization_transformed_images
            
            if (batch + 1) % verbose_hist_batch == 0:
                for name, param in weights.items():
                    writer.add_histogram('transform_net.'+name, param.clone().cpu().data.numpy(), 
                                         n_iter, bins='auto')
                
                for name, param in transform_net.named_parameters():
                    writer.add_histogram('transform_net.'+name, param.clone().cpu().data.numpy(), 
                                         n_iter, bins='auto')
                
                for name, param in metanet.named_parameters():
                    l = name.split('.')
                    l.remove(l[-1])
                    writer.add_histogram('metanet.'+'.'.join(l), param.clone().cpu().data.numpy(), 
                                         n_iter, bins='auto')

            pbar.set_description(s)
            
            del transformed_images, weights
        
    if not is_hvd or hvd.rank() == 0:
        torch.save(metanet.state_dict(), 'checkpoints/{}_{}.pth'.format(model_name, epoch+1))
        torch.save(transform_net.state_dict(), 
                   'checkpoints/{}_transform_net_{}.pth'.format(model_name, epoch+1))
        
        torch.save(metanet.state_dict(), 'models/{}.pth'.format(model_name))
        torch.save(transform_net.state_dict(), 'models/{}_transform_net.pth'.format(model_name))

