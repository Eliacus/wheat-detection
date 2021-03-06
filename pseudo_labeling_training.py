
"""
Pseudo labeling training script. Expects a pre-trained algorithm with decent accuracy. Assumed trained has been done for
100 epochs.
"""

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from util import *

DIR_INPUT = 'data'
DIR_TRAIN = f'{DIR_INPUT}/train'

# Load data and model

df = pd.read_csv('data/cleaned_data.csv')

weight_file = 'fasterrcnn_resnet50_fpn_2_remote.pth'

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

device = torch.device('cuda')

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(weight_file, map_location='cuda'))


# Pseudo labeling parameters
T1 = 100
T2 = 700
af = 3
detection_threshold = 0.5

image_ids = df['image_id'].unique()

val_size = int(np.round(0.2*len(image_ids)))

valid_ids = image_ids[-val_size:]
train_ids = image_ids[:-val_size]

valid_df = df[df['image_id'].isin(valid_ids)]
train_df = df[df['image_id'].isin(train_ids)]

train_dataset = WheatDataset(df, DIR_TRAIN, get_train_transforms())
pseudo_dataset = WheatDataset(df, DIR_TRAIN, get_train_transforms())

train_data_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

pseudo_data_loader = DataLoader(
    pseudo_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.00005, momentum=0.9, weight_decay=0.0005)

n_epochs = 100
batch_idx = 0
# Instead of using current epoch we use a "step" variable to calculate alpha_weight
# This helps the model converge faster
step = 101

train_loss = Averager()
pseudo_training_loss = Averager()
validation_loss = Averager()

for epoch in range(n_epochs):
    pseudo_training_loss.reset()
    train_loss.reset()
    for images_unlabeled, targets_unlabeled, image_ids_unlabeled in pseudo_data_loader:
        images_unlabeled = list(image.to(device) for image in images_unlabeled)

        # ------------- Predict Pseudo labels -----------------------
        model.eval()
        outputs = model(images_unlabeled)

        pseudo_targets = []
        for i, image in enumerate(images_unlabeled):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold]
            scores = scores[scores >= detection_threshold]
            if len(boxes) < 1:
                boxes = torch.tensor(([0, 0, 512, 512],), dtype=torch.float64)
                pseudo_targets.append({'boxes': boxes.to(device),
                                       'labels': torch.zeros((len(boxes),), dtype=torch.int64).to(device)})
                print("adding custom box")
            else:
                pseudo_targets.append({'boxes': torch.tensor(boxes, dtype=torch.float64).to(device),
                                       'labels': torch.ones(len(boxes), dtype=torch.int64).to(device)})

        # -----------------------------------------------------------------
        # Train on pseudo labels
        pseudo_targets = [{k: v.long().to(device) for k, v in t.items()} for t in pseudo_targets]
        model.train()
        loss_dict = model(images_unlabeled, pseudo_targets)

        alpha = alpha_weight(step, T1, T2, af)
        losses = alpha * sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        pseudo_training_loss.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        batch_idx += 1

        # For every 50 batches train one epoch on labeled data
        if batch_idx % 50 == 0:
            print(batch_idx, "batches complete. Loss: ", loss_value, " Training one epoch on labeled data")
            train_batch = 0
            for images, targets, image_ids in train_data_loader:
                train_batch += 1
                images = list(image.to(device) for image in images)
                targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                train_loss.send(loss_value)

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

                if train_batch % 50 == 0:
                    print(f"Training batch {train_batch} loss: {loss_value}")
            # Now we increment step by 1
            step += 1
            print("--------------------------------")
            print("Step ", step, " completed. Unlabeled loss: ", pseudo_training_loss.value,
                  ", Training loss: ", train_loss.value)
            print("--------------------------------")

            if not np.isnan(loss_value):
                print("Saving model")
                torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_2_pseudo_labeled.pth')

    print("Pseudo batch", epoch, " done")


