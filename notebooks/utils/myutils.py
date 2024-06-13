from datasets import load_dataset
from huggingface_hub.errors import HTTPError
import pandas as pd
import os

def get_full_ds_online(path: str):
    full_df = pd.DataFrame()
    try:
        full_ds = load_dataset('mcarthuradal/arm-unicef', name='full_features', split='train')
        full_df = full_ds.to_pandas()
        full_df = full_df.set_index('image_id')

        save_path = os.path.join(path, 'full_df.csv')
        full_df.to_csv(save_path, index=False)

    except (HTTPError, TypeError):
        print("Failed to get dataset from huggingface...")

    return full_df

def remove_tif_extension(filename):
  return os.path.basename(filename).split('.')[0]
    
    

def mend_objects(x: str):

  x= x.replace(r'array(', '') \
                .replace(r')', '') \
                .replace(r'dtype=object,', '') \
                .replace(r' ', '',  )  \
                .replace(r'\n', '',  )

  return eval(x)

def get_full_df_local(path: str):
    full_df = pd.read_csv(path)
    full_df.objects = full_df.objects.apply(mend_objects)

def remove_tif_extension(filename):
  return os.path.basename(filename).split('.')[0]

def draw_bbox(image_file: str, folder='train'):
    path = ''
    if not image_file.endswith(f'.{ext}'):
      path = os.path.join(f'images/{folder}', image_file + f'.{ext}')
    else:
      path = os.path.join(f'images/{folder}', image_file)
      image_file = remove_tif_extension(image_file)

    image = Image.open(path)
    image_id =  image_file
    row = full_df.loc[image_id]
    draw = ImageDraw.Draw(image)

    objects = row["objects"]
    categories = objects["category"]

    id2label = {
        '0': 'no object',
        '1': 'Other',
        '2': 'Tin',
        '3': 'Thatch'
    }
    label2id = {v: k for k, v in id2label.items()}

    for i in range(len(objects['bbox'])):
        box = objects["bbox"][i]
        annot_id = objects["id"][i]
        class_idx = str(objects["category"][i])
        area = objects['area'][i]
        x, y, w, h = tuple(box)
        # Check if coordinates are normalized or not
        # if max(box) > 1.0:
        # Coordinates are un-normalized, no need to re-scale them
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        draw.rectangle((x1, y1, x2, y2), outline="red", width=1)
        draw.text((x, y), id2label[class_idx]+f' {annot_id} ' + str(area), fill="white")
    return image
