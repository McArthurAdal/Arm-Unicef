# %%writefile submissions.py

from collections import Counter
import pandas as pd
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import v2
from argparse import ArgumentParser
import os
import torch
from PIL import Image
import tqdm

def load_model(args):
    model = fasterrcnn_resnet50_fpn(weights=None,
                                       weights_backbone='IMAGENET1K_V2',
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
    checkpoint = os.path.basename(checkpoint_path).split('.')[0]
    checkpoint = torch.load(checkpoint_path) 
    print(f"Successfully loaded checkpoint: epoch-{checkpoint['epoch']}, step-{checkpoint['global_step']}")
    print("Loading state_dict into model...")
    state_dict = {k.replace('model.', ''): v for k,v in checkpoint['state_dict'].items()}
    print("Done loading model...")
    model.load_state_dict(state_dict)
    return model
    

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

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--chkpt", type=str)
    args = parser.parse_args()
    
    model = load_model(args)
    model.to('cuda:0') 
    
    print("Predicting images in test folder...")
    
    submissions = []
    for image in tqdm(os.listdir('./images/test')):
      predict_image(f'images/test/{image}', model)
    
    print("Creating submissions dataframe...")
    
    submissions = pd.DataFrame(submissions)
    
    print("Writing submissions csv file to working folder")
    
    submissions.to_csv(f'submissions{chkpt}.csv', index=False)