import os
import numpy as np
import pandas as pd
from keras.utils import load_img, img_to_array
import concurrent.futures
from tqdm import tqdm

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG",
    ".ppm", ".PPM", ".bmp", ".BMP", ".tiff",
]

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)

def make_dataset(dir_path):
    images = []
    assert os.path.isdir(dir_path), f"{dir_path} is not a valid directory"
    for root, _, fnames in os.walk(dir_path):
        for fname in fnames:
            if is_image_file(fname):
                images.append(os.path.join(root, fname))
    return images

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def load_images(root_category, img_dir, target_size=(224, 224), max_workers=None, load_img_data=True, show_progress=False):
    """
    Load images from img_dir/root_category with parallel processing.
    Returns a dict with image arrays and metadata.
    
    Args:
        root_category: Root category folder name
        img_dir: Base directory for images
        target_size: Target size for images (width, height)
        max_workers: Maximum number of workers for parallel processing (None uses CPU count)
        load_img_data: Whether to load actual image data or just metadata
        show_progress: Whether to show a progress bar during loading
    """
    category_dir = os.path.join(img_dir, root_category)
    img_paths = make_dataset(category_dir)
    segmented_paths = [path.replace('\\','/').split('/') for path in img_paths]
    filenames = [seg[-1] for seg in segmented_paths]
    dates = [seg[-2] for seg in segmented_paths]
    labels = [seg[-3] for seg in segmented_paths]
    seq_nums = [fname.split('-')[0] for fname in filenames]
    seeds = [fname.split('-')[1].split('.')[0] for fname in filenames]
    ids = ['-'.join([d, s, sd]) for d, s, sd in zip(dates, seq_nums, seeds)]
    
    img_arrs = []
    if load_img_data:
        def load_single_image(path):
            try:
                img = load_img(path, target_size=target_size)
                return img_to_array(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return np.zeros((target_size[0], target_size[1], 3))
        
        paths_to_process = tqdm(img_paths, desc="Loading images") if show_progress else img_paths
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            img_arrs = list(executor.map(load_single_image, paths_to_process))
    
    return {
        "img_arrs": img_arrs,
        "img_paths": img_paths,
        "ids": ids,
        "filenames": filenames,
        "dates": dates,
        "labels": labels,
        "seq_nums": seq_nums,
        "seeds": seeds,
    }

def load_hierarchy(root_category, hierarchy_dir):
    file_path = os.path.join(hierarchy_dir, f"{root_category.casefold()}.csv")
    return pd.read_csv(file_path)

def create_metadata_df(image_data, hierarchy_df):
    metadata_df = pd.DataFrame(
        list(zip(
            image_data["ids"], 
            image_data["labels"], 
            image_data["seq_nums"], 
            image_data["seeds"], 
            image_data["dates"], 
            image_data["filenames"], 
            image_data["img_paths"]
        )),
        columns=['id', 'class_name', 'sequence_number', 'seed', 'date_created', 'filename', 'path']
    )
    
    metadata_df = metadata_df.reset_index().merge(
        hierarchy_df, left_on='class_name', right_on='class', suffixes=['','_y']
    )
    
    return metadata_df[['class', 'cat_depth_0', 'cat_depth_1', 'cat_depth_2', 'frequency']].fillna('None')
