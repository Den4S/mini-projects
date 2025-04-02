"""
Module to download and process Weizmann dataset and/or masks.
    Source of the code: https://github.com/ztangent/multimodal-dmm/blob/master/datasets/weizmann.py
"""

import os, sys
import requests
from tqdm import tqdm
import zipfile

import numpy as np

import scipy.io
import torchvision

import skimage.transform

# import skvideo
# skvideo.setFFmpegPath('/Users/giyuu/science-phd/git-projects/SVETlANNa.docs/venv/bin/')
# import skvideo.io


PERSONS = [
    'daria', 'denis', 'eli', 'ido', 'ira',
    'lena', 'lyova', 'moshe', 'shahar'
]
ACTIONS = [
    'bend', 'jack', 'jump', 'pjump', 'run',
    'side', 'skip', 'walk', 'wave1', 'wave2'
]
DESCRIPTIONS = [
    'Bend', 'Jumping jack', 'Jump',
    'Jump in place', 'Run' 'Gallop sideways',
    'Skip', 'Walk', 'One-hand wave', 'Two-hand wave'
]

DUPLICATES = ['lena_walk', 'lena_run', 'lena_skip']  # there are two files for each


def download_weizmann_masks(dest='./weizmann'):
    """
    Downloads masks for Weizmann human action dataset.
    """
    
    src_url = ('http://www.wisdom.weizmann.ac.il/~vision/' +
               'VideoAnalysis/Demos/SpaceTimeActions/DB/')

    # Download masks / silhouettes
    print("Downloading masks...", end=' ')
    
    if not os.path.exists(dest):
        os.mkdir(dest)
    if not os.path.exists(os.path.join(dest, 'classification_masks.mat')):
        download('classification_masks.mat', source=src_url, dest=dest)
        
    masks = scipy.io.loadmat(os.path.join(dest, 'classification_masks.mat'))
    masks = masks['original_masks'][0, 0]
    
    print('Done!')  # masks downloaded!

    for act in ACTIONS:
        # Download masks for each action
        
        vid_names_no_ext = [f'{person}_{act}' for person in PERSONS]

        path_act_masks = os.path.join(dest, f'{act}_masks')  # folder for action masks
        print(f"Saving masks for `{act}` as .npy to `{path_act_masks}`...", end=' ')

        for vn_no_ext_0 in vid_names_no_ext:
            if not os.path.exists(path_act_masks):
                os.mkdir(path_act_masks)
            
            npy_path = os.path.join(path_act_masks, vn_no_ext_0 + '.npy')

            if not os.path.exists(npy_path):  # if file is not exist
                # Remove extension
                vn_no_ext = vn_no_ext_0
    
                if vn_no_ext in DUPLICATES:
                    # Skip duplicate videos (e.g. 'lena_walk2.avi')
                    vn_no_ext += '1'
    
                try:  # check if key exists!
                    mask_data = preprocess_mask(masks[vn_no_ext])
                except ValueError:
                    continue
                                
                np.save(npy_path, mask_data)

        print('Done!')


def preprocess_mask(mask):
    """
    Crop, normalize and swap dimensions.
    """
    height, width = mask.shape[0:2]
    side = min(height, width)
    x0 = (width - side)//2
    y0 = (height - side)//2
    # Crop to central square, convert to float
    mask = np.array(mask[y0:y0+side, x0:x0+side, :]).astype(np.float64)
    # Transpose to (time, rows, cols)
    mask = np.transpose(mask, (2,0,1))
    # Resize to 64 by 64
    mask = np.stack(
        [skimage.transform.resize(mask[t], (64, 64))
        for t in range(mask.shape[0])], axis=0
    )  # PROBLEMS MAY OCCURE THERE! CHECK IT!
    
    # Add channels dimension
    mask = mask[:, np.newaxis, :, :]
    return mask


def download(filename, source, dest):
    print("Downloading '{}'...".format(filename))
    
    url = source + filename
    
    try:

        with open(os.path.join(dest, filename), 'ab') as f:
            headers = {}
            pos = f.tell()
            if pos:
                headers['Range'] = 'bytes={}-'.format(pos)
            resp = requests.get(url, headers=headers, stream=True)
            total_size = resp.headers.get('content-length', None)
            total = int(total_size)//1024 if total_size else None
            for data in tqdm(iterable=resp.iter_content(chunk_size=512),
                             total=total, unit='KB'):
                f.write(data)

    except requests.exceptions.RequestException:

        print("\nError downloading, attempting to resume...")
        download(filename, source, dest)

