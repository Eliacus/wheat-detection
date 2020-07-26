
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import torch
import torchvision
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from matplotlib import pyplot as plt

from util import *

DIR_INPUT = 'data'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

# Create pandas dataframe
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

# Expand bboxes to x,y,w,h
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

# train_df.drop(columns=['bbox'], inplace=True)

train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

train_df['area'] = train_df['w'] * train_df['h']


# Histogram over area distribution
# plt.hist(train_df['area'])

train_df.drop(train_df[train_df['area'] > 200000].index, inplace=True)
large_boxes = train_df[train_df['area'] > 200000].image_id

train_df.to_csv('data/cleaned_data.csv')
# plot_image_examples(train_df[train_df.image_id.isin(large_boxes)])
