if __name__ == '__main__':
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from util import *

    DIR_INPUT = 'data'
    DIR_TRAIN = f'{DIR_INPUT}/train'
    DIR_TEST = f'{DIR_INPUT}/test'

    # Create pandas dataframe
    train_df = pd.read_csv(f'{DIR_INPUT}/cleaned_data.csv')

    image_ids = train_df['image_id'].unique()

    # val_size = int(np.round(0.2*len(image_ids)))
    #
    # valid_ids = image_ids[-val_size:]
    # train_ids = image_ids[:-val_size]
    #
    # valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    # train_df = train_df[train_df['image_id'].isin(train_ids)]

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the trained weights
    # weight_file = 'fasterrcnn_resnet50_fpn_3.pth'
    # model.load_state_dict(torch.load(weight_file, map_location='cuda'))

    train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transforms())
    # valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transforms())

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # valid_data_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=8,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=collate_fn
    # )

    device = torch.device('cuda')

    # ------------------------------------- Training -------------------------------------

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    lr_scheduler = None

    num_epochs = 100

    train_loss = Averager()
    # validation_loss = Averager()
    itr = 1

    for epoch in range(num_epochs):
        train_loss.reset()

        for images, targets, image_ids in train_data_loader:

          #  print(np.max(targets[0]['boxes'].cpu().numpy()),
          #        np.max(targets[1]['boxes'].cpu().numpy()),
          #        np.max(targets[2]['boxes'].cpu().numpy()),
          #        np.max(targets[3]['boxes'].cpu().numpy()))
          #  print(np.min(targets[0]['boxes'].cpu().numpy()),
          #        np.min(targets[1]['boxes'].cpu().numpy()),
          #        np.min(targets[2]['boxes'].cpu().numpy()),
          #        np.min(targets[3]['boxes'].cpu().numpy()))

            images = list(image.to(device) for image in images)

            targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            train_loss.send(loss_value)

            if np.isnan(loss_value):
                print(targets)

            optimizer.zero_grad()
            losses.backward()

            optimizer.step()

            if itr % 1 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1

        print(f"Epoch #{epoch} training loss: {train_loss.value}")

        # with torch.no_grad():
        #     for images, targets, image_ids in valid_data_loader:
        #         images = list(image.to(device) for image in images)
        #         targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]
        #         loss_dict = model(images, targets)
        #
        #         losses = sum(loss for loss in loss_dict.values())
        #         loss_value = losses.item()
        #
        #         validation_loss.send(loss_value)

        # print(f"Epoch #{epoch} loss: {validation_loss.value}")

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        torch.save(model.state_dict(), 'cropping.pth')

    # ------------------------------------- Validation -------------------------------------

    torch.save(model.state_dict(), 'crop_final.pth')

