
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
train_df = pd.read_csv(f'{DIR_INPUT}/cleaned_data.csv')

image_ids = train_df['image_id'].unique()

val_size = int(np.round(0.2*len(image_ids)))

valid_ids = image_ids[-val_size:]
train_ids = image_ids[:-val_size]

valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transforms())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transforms())

# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

device = torch.device('cuda')

# ------------------------------------- Sample -------------------------------------
sampling = False
if sampling:
    images, targets, image_ids = next(iter(train_data_loader))
    images = list(image.to(device) for image in images)
    targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]

    boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
    sample = images[0].permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 3)

    ax.set_axis_off()
    ax.imshow(sample)
    plt.show()


# ------------------------------------- Training -------------------------------------

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 50

loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()

    for images, targets, image_ids in train_data_loader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")
    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')

# ------------------------------------- Validation -------------------------------------

# test_id = 5
# images, targets, image_ids = next(iter(valid_data_loader))
# images = list(img.to(device) for img in images)
#
# targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]
#
# boxes = targets[test_id]['boxes'].cpu().numpy().astype(np.int32)
# sample = images[test_id].permute(1, 2, 0).cpu().numpy()
#
# model.eval()
# cpu_device = torch.device("cpu")
#
# outputs = model(images)
# outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#
# fig, ax = plt.subplots(1, 1, figsize=(16, 8))
#
# for box in boxes:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (0, 220, 0), 3)
#
# for box in outputs[test_id]['boxes']:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 3)
#
# ax.set_axis_off()
# ax.imshow(sample)

torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')

