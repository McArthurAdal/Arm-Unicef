# %% [code]
# %%writefile submissions.py

from collections import Counter
import pandas as pd
from torchvision.io.image import read_image
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          fasterrcnn_resnet50_fpn_v2,
                                          fasterrcnn_mobilenet_v3_large_fpn
                                          )
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import v2
from argparse import ArgumentParser
import os
import torch
from PIL import Image
from tqdm import tqdm

def load_model(args):
    model = fasterrcnn_resnet50_fpn_v2(weights=None,
                                       weights_backbone='DEFAULT',
                                       num_classes=4,
                                       max_size = 1000,
                                       min_size= 500,
                                       image_mean = [0.4794, 0.4564, 0.3860],
                                       image_std =  [0.1953, 0.1782, 0.1870],
                                      #  trainable_backbone_layers=1,
                                     # inference 
                                       rpn_pre_nms_top_n_train=1000,
                                       rpn_pre_nms_top_n_test=50,
                                       rpn_post_nms_top_n_train=500,
                                       rpn_post_nms_top_n_test=50,
                                    )
    print("Loading model from checkpoint...")
    checkpoint_path = f'{args.chkpt}'
    checkpoint_name, ext = os.path.basename(checkpoint_path).split('.') 
    checkpoint = torch.load(checkpoint_path) 
    print(f"Successfully loaded checkpoint: {checkpoint_name}")
    print("Loading state_dict into model...")
    if ext == 'ckpt':
        print('Loading Lightning style checkpoint')
        state_dict = {k.replace('model.', ''): v for k,v in checkpoint['state_dict'].items()}
        print("Done loading model...")
        model.load_state_dict(state_dict)
        return model, checkpoint_name
    
    elif ext == 'pth':
        print('Loading PyTorch style checkpoint')
        state_dict =  checkpoint
        print("Done loading model...")
        model.load_state_dict(state_dict)
        return model, checkpoint_name
    

def predict_image(image_path, model):

      model.eval()

      # Step 2: Initialize the inference v2
      preprocess = v2.Compose([
          v2.PILToTensor(),
          v2.ToImage(),
          v2.ToDtype(torch.float32, scale=True)
      ])

      with Image.open(image_path) as img:
        batch = [preprocess(img).cuda()]

      # Step 4: Use the model and visualize the prediction
      prediction = model(batch)[0]
      idx = prediction["labels"].tolist()

      counts = Counter(idx)

      image_id = os.path.basename(image_path).split('.')[0]

      submissions.append({'image_id': f'{image_id}_{1}', 'Target': counts[1]})
      submissions.append({'image_id': f'{image_id}_{2}', 'Target': counts[2]})
      submissions.append({'image_id': f'{image_id}_{3}', 'Target': counts[3]})
      
#       boxes = prediction['boxes']
#       scores = prediction['scores']
#       labels = []
#       id2label = {
#                     0: 'no object',
#                     1: 'Other',
#                     2: 'Tin',
#                     3: 'Thatch'
#                 }
#       for label in idx:
#           labels.append(id2label[label])
      
#       pred_targets.append({'image_id': image_id, 'boxes': boxes, 'labels': labels, 'scores': scores})
#       del prediction, batch
#       torch.cuda.empty_cache()
        
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--chkpt", type=str)
    args = parser.parse_args()
    
    model, checkpoint_name = load_model(args)
    model.to('cuda:0') 
    
    print("Predicting images in test folder...")
    
    submissions = []
    pred_targets = []
    for image in tqdm(os.listdir('./images/test')):
      predict_image(f'images/test/{image}', model)
      
    
    print("Creating submissions dataframe...")
    
    submissions = pd.DataFrame(submissions)
#     predictions = pd.DataFrame(pred_targets)
    model_name = model.__class__.__name__
    os.makedirs('submissions', exist_ok=True)
    print("Writing submissions csv file to working folder...\nDone!")
    submissions.to_csv(f'{model_name}_submissions{checkpoint_name}.csv', index=False)
#     predictions.to_csv(f'{model_name}_predictions{checkpoint_name}.csv', index=False)