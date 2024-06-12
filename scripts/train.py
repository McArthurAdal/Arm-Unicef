%%writefile train.py

# %% [code] {"executionInfo":{"elapsed":4083,"status":"ok","timestamp":1716811785558,"user":{"displayName":"Mark Kyama","userId":"17617811822804530741"},"user_tz":-180},"id":"rDzuSIGa1QOl","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-08T01:56:04.787479Z","iopub.execute_input":"2024-06-08T01:56:04.787773Z","iopub.status.idle":"2024-06-08T01:56:08.419996Z","shell.execute_reply.started":"2024-06-08T01:56:04.787749Z","shell.execute_reply":"2024-06-08T01:56:08.419021Z"}}
import os
from huggingface_hub import HfApi, login, hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F
import pynvml
import torch
import logging

# Configure the logger
logging.basicConfig(filename='validation_metrics.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# %% [code] {"executionInfo":{"elapsed":4310,"status":"ok","timestamp":1716811789864,"user":{"displayName":"Mark Kyama","userId":"17617811822804530741"},"user_tz":-180},"id":"myFCODpLpU33","outputId":"841696e7-6e0e-4eb0-ba30-54200c8bd939","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-08T01:56:08.421654Z","iopub.execute_input":"2024-06-08T01:56:08.422063Z","iopub.status.idle":"2024-06-08T01:56:08.735251Z","shell.execute_reply.started":"2024-06-08T01:56:08.422035Z","shell.execute_reply":"2024-06-08T01:56:08.734227Z"}}
# from kaggle_secrets import UserSecretsClient

# secret_label = "HF_TOKEN"
# HF_TOKEN = UserSecretsClient().get_secret(secret_label)

# login(token=HF_TOKEN)

arm_unicef_folder = '/kaggle/input/arm-unicef-disaster-vulnerability-challenge'
local_dir = 'kaggle/working' 

import pandas as pd
import numpy as np
train_csv = os.path.join(arm_unicef_folder,  'Train.csv' )
test_csv = os.path.join(arm_unicef_folder,   'Test.csv' )

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-08T01:56:08.736619Z","iopub.execute_input":"2024-06-08T01:56:08.737137Z","iopub.status.idle":"2024-06-08T01:56:08.799301Z","shell.execute_reply.started":"2024-06-08T01:56:08.737102Z","shell.execute_reply":"2024-06-08T01:56:08.798375Z"}}
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-08T01:57:35.142727Z","iopub.execute_input":"2024-06-08T01:57:35.143089Z","iopub.status.idle":"2024-06-08T01:57:41.283451Z","shell.execute_reply.started":"2024-06-08T01:57:35.143059Z","shell.execute_reply":"2024-06-08T01:57:41.282642Z"}}
images_folder = '/kaggle/working/images/'

train_folder = os.path.join(images_folder, 'train')
test_folder = os.path.join(images_folder, 'test')

from datasets import load_dataset
from huggingface_hub.errors import HTTPError
import pandas as pd

full_df = pd.DataFrame()
local_fulldf = False
try:
  full_ds = load_dataset('mcarthuradal/arm-unicef', name='full_features', split='train')
except (HTTPError, TypeError):
  full_df = pd.read_csv(f'{arm_unicef_folder}/train/full_df.csv')
  local_fulldf = True

from pprint import pprint

def mend_objects(x: str):

    x = x.replace(r'array(', '') \
                .replace(r')', '') \
                .replace(r'dtype=object,', '') \
                .replace(r' ', '',  )  \
                .replace(r'\n', '',  )
    return eval(x)

if not local_fulldf:
    full_df = full_ds.to_pandas()
    full_df = full_df.set_index('image_id')
else:
    full_df.objects = full_df.objects.apply(mend_objects)

full_df.image = full_df.image.str.replace('jpeg', 'tif')

def remove_extension(filename):
  return re.sub(r'\.tif$', '', filename)

from PIL import Image, ImageDraw
import re

def to_numpy(objects):
  boxes = objects['bbox']
  categories = objects['category']

  b = []
  for box in boxes:
    a = [int(coord) for coord in box]
    b.append(a)

  new_boxes = np.array(b, dtype=np.int32)
  new_categories = np.array(categories, dtype=np.int64)

  objects['bbox'] = new_boxes
  objects['category'] = new_categories

  return objects

try:
  full_df.objects = full_df.objects.apply(to_numpy)
except TypeError:
  print("Error")

with open('/kaggle/input/good-bad-images/bad.txt', 'r') as f:
  bad_files = f.readlines()
  bad_files = [x.strip() for x in bad_files]

full_df['good'] = ~full_df.index.isin(bad_files)


import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

class CustomDataset(VisionDataset):
    def __init__(self, root, df, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.df = df.copy().query('good==True')



    def __getitem__(self, index):
        filename = self.df.iloc[index]['image']
        path = os.path.join(train_folder, filename)
        ids = self.df.iloc[index]['objects']['id']
        # make sure to apply custom func to_numpy on df
        target = {'bbox': torch.IntTensor(self.df.iloc[index]['objects']['bbox']),
                  'label': torch.IntTensor(self.df.iloc[index]['objects']['category'])
                  }

        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            size = self.get_image_size(index)
            new_size = size.expand(target['bbox'].shape[0],size.numel())
            # target['bbox'] = self.target_transform(zip(target['bbox'], new_size, target['label']))['bbox']
            target['bbox'] = self.target_transform(zip(target['bbox'], new_size))

        return (image, target, filename, ids)

    def get_image_size(self, index):
          w,h = self.df.iloc[index]['width'], self.df.iloc[index]['height']
          return torch.Tensor([w, h])

    def __len__(self):
      return len(self.df)

import torch
from torchvision import tv_tensors

def to_bbox(zipped):

  boxes = []
  size = torch.Tensor([0, 0])
    
  # for tgts, sizes, lbls in zipped:
  for tgts, sizes  in zipped:
    # append individual boxes for a single image
    boxes.append(tgts)
    size = sizes


  boxes = torch.stack(boxes)
  
  w, h = int(size[0].item()), int(size[1].item())

  return tv_tensors.BoundingBoxes(boxes ,
                           canvas_size= (w,h),
                           dtype= torch.int64,
                           format=tv_tensors.BoundingBoxFormat.XYWH)
def convert_to_list_of_dicts(batch_bboxes, batch_labels):

  results = []
  for i in range(batch_bboxes.shape[0]):
    bboxes = batch_bboxes[i]
    labels = batch_labels[i]

    # Create a dictionary for the current image
    result = {
      "boxes": bboxes,
      "labels": labels
    }

    results.append(result)

  return results

from itertools import islice
from torch.utils.data import DataLoader, random_split


def custom_collate_fn(batch: list):


  # Get the maximum number of bounding boxes and labels across all images in the batch.
  max_num_boxes = max([len(sample['bbox']) for _, sample, _, _ in batch])
  max_num_labels = max([len(sample['label']) for _, sample, _, _ in batch])

  # Create empty tensors for images, bounding boxes, and labels.
  # images = torch.empty((len(batch), 3, 500, 500))
  images = []
  bboxes = torch.zeros((len(batch), max_num_boxes, 4) )
  labels = torch.zeros((len(batch), max_num_labels), dtype=torch.int64)
  img_ids = []

  # Fill in the tensors with data from each sample.
  for i, sample in enumerate(batch):
      images.append(sample[0])#.to(device))
      bboxes[i, :len(sample[1]['bbox'])]  = sample[1]['bbox']
      labels[i, :len(sample[1]['label'])] = sample[1]['label']

      img_ids.append(sample[2])

  targets = convert_to_list_of_dicts(bboxes,labels)

  return images, targets, img_ids

import torchvision
import torch.nn as nn
from torchvision.io.image import read_image
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          fasterrcnn_resnet50_fpn_v2,
                                          FasterRCNN_ResNet50_FPN_Weights,
                                          FasterRCNN_ResNet50_FPN_V2_Weights)

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


def init_model(args):
    model = fasterrcnn_resnet50_fpn(weights=None,
                                   weights_backbone='IMAGENET1K_V2',
                                   num_classes=4,
                                   max_size = 1000,
                                   min_size= 500,
                                   image_mean = [0.4794, 0.4564, 0.3860],
                                   image_std =  [0.1953, 0.1782, 0.1870],
                                  #  trainable_backbone_layers=1,
                                   rpn_pre_nms_top_n_train=1000,
                                   rpn_pre_nms_top_n_test=50,
                                   rpn_post_nms_top_n_train=500,
                                   rpn_post_nms_top_n_test=50,
                                  )
    if not args.newmodel:
        print('Initializing pretrained model from checkpoint')

        checkpoint_path = '/kaggle/working/mAP/epoch=4-step=210.ckpt'

        checkpoint = torch.load(checkpoint_path)
         
        state_dict = {k.replace('model.', ''): v for k,v in checkpoint['state_dict'].items()}

        model.load_state_dict(state_dict)

        print('Model state dict loaded')
        return model
    else:
        print('Working with new pretrained model')
        return model


import torch
from torchvision import datasets as torchds, models
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch.nn as nn
import torch.optim as optim


policy = v2.AutoAugmentPolicy.IMAGENET
transform_n = v2.Compose([
    v2.PILToTensor(),
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.RandomHorizontalFlip(p=1),
    v2.RandomRotation(degrees=(0,180)),
#     v2.AutoAugment(policy),
    v2.RandomInvert(p=0.5), 
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
])

target_tf2 = v2.Compose([v2.Lambda(lambda x : to_bbox(x) ),
                         v2.ConvertBoundingBoxFormat('XYXY')
                        ])


def get_loader(batch_size=32, tranform_n=transform_n):
      train_dataset_n = CustomDataset(train_folder, full_df.sort_values(by='width'), transform_n, target_tf2)
      train_set_n, val_set_n = random_split(train_dataset_n, [0.8, 0.2])

      train_loader_n = DataLoader(train_set_n ,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=custom_collate_fn,

                                   )

      val_loader_n = DataLoader(val_set_n, sampler=None, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

      return train_loader_n, val_loader_n

from  torchvision.tv_tensors import BoundingBoxes

def targets_to_bbox(targets: list[torch.IntTensor]):
  bb = BoundingBoxes(targets['boxes'], format='XYXY', canvas_size=(500,500))
  ll = targets['labels']

  return {'boxes': bb , #.to(device),
          'labels': ll  #.to(device)
         }

clean_tf = v2.Compose([v2.SanitizeBoundingBoxes(),
                         v2.ToPureTensor(),
                         v2.ToDtype(torch.int64 )
                        ])
def train_one_batch(*, model, images, targets):

    # sanitize boxes for 1 or N images
    targets_clean = []
    for pair in targets:
        targets_clean.append(clean_tf(targets_to_bbox(pair)) )

    outputs = model(images, targets_clean)
    loss = combined_loss(outputs)
#     loss.backward()
#     optimizer.step()
    return loss#.item()


def validate_model(model, images, targets):

    targets_clean = []
    for pair in targets:
        targets_clean.append(clean_tf(targets_to_bbox(pair)) )

    outputs = model(images, targets_clean)
#     loss = combined_loss(outputs)

    return outputs 



def combined_loss(outputs): 
    # RCNN
    lambda_loc = 1
    lambda_cls = 1
    #RPN
    lambda_obj = 1
    lambda_rpn = 1
    
    loc_loss = outputs['loss_box_reg']
    cls_loss = outputs['loss_classifier']
    obj_loss = outputs['loss_objectness']
    rpn_loss = outputs['loss_rpn_box_reg']

    total_loss = lambda_loc * loc_loss  + lambda_cls * cls_loss   + lambda_obj * obj_loss + lambda_rpn * rpn_loss
    return total_loss

from tqdm import tqdm
import mlflow
import lightning as L
from torchmetrics.detection  import  mean_ap, iou, giou, ciou

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-08T01:57:46.344429Z","iopub.execute_input":"2024-06-08T01:57:46.344788Z","iopub.status.idle":"2024-06-08T01:57:46.355469Z","shell.execute_reply.started":"2024-06-08T01:57:46.344760Z","shell.execute_reply":"2024-06-08T01:57:46.354550Z"}}
from lightning.pytorch import Trainer, seed_everything

seed_everything(42, workers=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-08T01:58:18.243634Z","iopub.execute_input":"2024-06-08T01:58:18.243978Z","iopub.status.idle":"2024-06-08T01:58:18.263788Z","shell.execute_reply.started":"2024-06-08T01:58:18.243941Z","shell.execute_reply":"2024-06-08T01:58:18.262825Z"}}
class LightningFRCNN(L.LightningModule):
    
    def __init__(self, model, batch_size, lr=0.001):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.loader = get_loader(batch_size)
        self.mAP = mean_ap.MeanAveragePrecision(max_detection_thresholds=[1,50,100], class_metrics=True, backend='faster_coco_eval', extended_summary=True)  
        self.iou = iou.IntersectionOverUnion(class_metrics=True, respect_labels=True)
        self.ciou = ciou.CompleteIntersectionOverUnion(class_metrics=True, respect_labels=True )
        self.giou = giou.GeneralizedIntersectionOverUnion(class_metrics=True, respect_labels=True)
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, images, targets):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        loss = train_one_batch(model=self.model,images=images, targets=targets)
        self.log("train_loss", 
                 loss, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)
        
        self.log('batch_size', self.batch_size, batch_size=self.batch_size)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        preds = validate_model(model=self.model, images=images, targets=targets)
        
        mAP = self.mAP(preds, targets)
        iou = self.iou(preds, targets)
        giou = self.giou(preds, targets)
        
        self.log('mAP', mAP['map'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('IoU', iou['iou'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('GIoU', giou['giou'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return iou['iou']
    
    def test_step(self, batch, batch_idx):
        images = batch
        outputs = self.model(images)
        return outputs
    
    def configure_optimizers(self):
#         parameters = [model.named_parameters() for model.named_parameters() in ]
        return optim.Adam(self.model.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        train_l =  self.loader[0]
        return train_l
    
    def val_dataloader(self):
        val_l = self.loader[1]
        return val_l
    
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, StochasticWeightAveraging, EarlyStopping

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

early_stopping = EarlyStopping('train_loss')
mchkpt = ModelCheckpoint('train_loss')

from lightning.pytorch.loggers import MLFlowLogger

def main(args):
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
    
    model = init_model(args)
    LightningModel = LightningFRCNN(model, batch_size=args.batch_size)

    trainer = L.Trainer(deterministic=False,
                       accelerator='auto',
                       devices='auto',
                       strategy='ddp',
                       use_distributed_sampler=True,
                       callbacks=[PrintCallback(),
                                 StochasticWeightAveraging(swa_lrs=1e-2),
                                 early_stopping,
                                  mchkpt
                                 ],
                       default_root_dir=os.getcwd(),
                       enable_checkpointing=True,
                       fast_dev_run=args.fast_dev_run, # Debugger
                       logger=mlf_logger, # TensorBoard
                       max_epochs=100,
                       max_time="00:12:00:00",
                       profiler= None,
                       inference_mode=False,# Use `torch.no_grad` instead
    #                    limit_train_batches=0.1,
    #                    limit_val_batches=0.01
                       )

    trainer.fit(model=LightningModel)
 
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--fast_dev_run", type=int) 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--newmodel", action='store_true')


    args = parser.parse_args()
    main(args)

# !rm -rf state_dicts train_loss ml-runs state.db

# %% [code] {"execution":{"iopub.status.busy":"2024-06-08T02:18:01.159630Z","iopub.execute_input":"2024-06-08T02:18:01.160016Z","iopub.status.idle":"2024-06-08T02:18:01.268942Z","shell.execute_reply.started":"2024-06-08T02:18:01.159981Z","shell.execute_reply":"2024-06-08T02:18:01.267725Z"}}

# t_model = LightningFRCNN .load_from_checkpoint('/kaggle/working/train_loss/epoch=19-step=840.ckpt')
# t_model.eval()
# t_model()

# import random
# import glob

# train_images = glob.glob(f'./images/train/*.tif')

# random_image = random.sample(train_images, 1)

# print(random_image)

# from torchvision.io.image import read_image
# from torchvision.utils import draw_bounding_boxes
# from torchvision.transforms.functional import to_pil_image



# model.eval()

# # Step 2: Initialize the inference v2
# preprocess = v2.Compose([
#     v2.PILToTensor(),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True)
# ])

# preprocess2 = v2.Compose([
#     v2.PILToTensor(),
#     v2.ToImage(),
     
# ])

# with Image.open(random_image[0]) as img:

#     batch = [preprocess(img).cuda()]

# # Step 4: Use the model and visualize the prediction
# prediction = model(batch)[0]
# idx = prediction["labels"].tolist()
# boxes = prediction["boxes"]

# labels = []

# for label in idx :
#   id2label = {
#         0: 'no object',
#         1: 'Other',
#         2: 'Tin',
#         3: 'Thatch'
#     }
#   labels.append(id2label[label])
# to_pil_image(draw_bounding_boxes(preprocess2(img), boxes=boxes,labels=labels, colors='red'))
