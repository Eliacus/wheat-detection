from torch.utils.data import DataLoader

from util import *
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


if __name__ == '__main__':
    test_df = pd.read_csv('data/sample_submission.csv')
    weight_file = 'fasterrcnn_resnet50_fpn_2.pth'

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    device = torch.device('cpu')

    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(weight_file, map_location='cpu'))
    model.eval()

    x = model.to(device)

    test_dataset = WheatTestDataset(test_df, 'data/test', get_test_transforms())

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn
    )

    detection_threshold = 0.92
    results = []
    plot_df = pd.DataFrame()
    for images, image_ids in test_data_loader:

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()
            for j, box in enumerate(boxes):
                if scores[i] > detection_threshold:
                    plot_df = plot_df.append({'image_id': image_ids[i], 'x': box[0], 'y': box[1]
                                              , 'w': box[2] - box[0], 'h': box[3] - box[1]}, ignore_index=True)

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }

            results.append(result)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    print(test_df.head())
    test_df.to_csv('submission.csv', index=False)
    plot_image_examples(plot_df, rows=2, cols=2)

