import cv2
import argparse
import logging
import os
import sys
from collections import OrderedDict
from typing import Optional
import torch
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor

from demo.predictor import VisualizationDemo
device = torch.device("cuda")
def setup_cfg(config_file, opts, confidence_threshold):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)

    # -----
    from SwinTextSpotter.projects.SWINTS.swints import add_SWINTS_config
    add_SWINTS_config(cfg)
    # -----

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.SWINTS.PATH_COMPONENTS = "SwinTextSpotter/projects/SWINTS/LME/coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60_siz28.npz"
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg


input_img_path = "test_img/392287007.png"
img_orig = cv2.imread(input_img_path)
# img_orig = cv2.resize(img_orig, [1000, 1000], interpolation = cv2.INTER_CUBIC)

config_file = "SwinTextSpotter/projects/SWINTS/configs/SWINTS-swin-pretrain.yaml"
opts = ['MODEL.WEIGHTS', 'trained_models/swin_imagenet_pretrain.pth']

confidence_threshold = 0.2
cfg = setup_cfg(config_file, opts, confidence_threshold)

model = build_model(cfg).eval()


print(list(model.parameters())[0].shape, list(model.parameters())[1].shape, list(model.parameters())[2].shape )

checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)

print([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

input_format = cfg.INPUT.FORMAT
assert input_format in ["RGB", "BGR"], input_format

"""
Args:
    original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

Returns:
    predictions (dict):
        the output of the model for one image only.
        See :doc:`/tutorials/models` for details about the format.
"""
with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
    # Apply pre-processing to image.
    if input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        original_image = img_orig[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    nH, nW = image.shape[:2]
    # print(nH, nW)

    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).unsqueeze(0).to(device)

    # inputs = {"image": image, "height": torch.tensor(height), "width": torch.tensor(width)}
    
    # print(image.shape, height, width)
    predictions = model([{"image": image}])[0]

# pred = DefaultPredictor(cfg)
# inputs = cv2.imread(input_img_path)
# predictions = pred(inputs)

instances = predictions["instances"]
instances = instances[instances.scores > confidence_threshold]
outputs = instances.pred_boxes.tensor.detach().cpu().numpy()
print(outputs.shape)

## plot bbox
save_path = "results_1.jpg"
def plot(image, boxes, save_path):
    #image in the form of cv2.imread()
    # gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    N, _ = boxes.shape
    for i in range(N):
        print(int(boxes[i][0]*width/nW), int(boxes[i][1]*height/nH), int(boxes[i][2]*width/nW), int(boxes[i][3]*height/nH))
        color = [100, 50, 100]
        # print(box)
        cv2.rectangle(image, (int(boxes[i][0]*width/nW), int(boxes[i][1]*height/nH)), (int(boxes[i][2]*width/nW), int(boxes[i][3]*height/nH)), color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.imwrite(save_path, image)

plot(cv2.imread(input_img_path), outputs, save_path)



