import os
import json
from math import ceil
import random

def add_dim(src):
    for social in os.listdir(src):
        social_path = os.path.join(src, social)
        for vid in os.listdir(social_path):
            vid_path = os.path.join(social_path, vid)
            vid_data = json.load(open(vid_path))
            last_mb = vid_data['frames'][0]['macroblocks'][-1]
            vid_data['x'], vid_data['y'] = last_mb['x'], last_mb['y']
            with open(vid_path, 'w') as outfile:
                json.dump(vid_data, outfile)

def create_train_test_split(path, amount=.8):
    train = list()
    test = list()

    for fld in os.listdir(path):
        full = [f'{fld}_{os.path.splitext(vid)[0]}' for vid in os.listdir(os.path.join(path, fld))]
        random.shuffle(full)
        num_train_samples = ceil(len(full) * amount)
        train += full[:num_train_samples]
        test += full[num_train_samples:]
    
    with open(os.path.join(path, 'train.txt'), 'w') as fp:
        for item in train:
            # write each item on a new line
            fp.write(f"{item}\n")
    
    with open(os.path.join(path, 'test.txt'), 'w') as fp:
        for item in test:
            # write each item on a new line
            fp.write(f"{item}\n")

create_train_test_split('videos/v3')