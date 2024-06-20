from pynvml import *

auth_cred_fpath = "auth.json" # user credentials
auth_message_fpath = "data_agreement.html" # data agreement file path
user_data_store_fpath = "user-data/public_user_data.csv" 

SAFETY_CHECK = True

CRED_FPATH = "auth.json"
MODEL_NAME_PATH_MAP = {
    "Oil Painting": "models/oilPaintingV10.safetensors",
    "Watercolor": "models/colorwater-v4.safetensors", 
    "MoXin (traditional Chinese Painting)": "models/MoXin-v1.safetensors",
    "Architectural Sketch": "models/architectural_sketch.safetensors",
    "Beautiful Outdoor Background": "models/background_beautiful_outdoor.safetensors",
    "Fashion Cloth": "models/fashion_cloth.safetensors", 
    "Pencil Sketch": "models/pencil_sketch.safetensors",
    "Slate Pencil Mix": "models/slate_pencil_mix.safetensors",
}

IMAGE_STYLE_CHOICES = ["Oil Painting", 
                       "Watercolor", 
                       "MoXin (traditional Chinese Painting)",]
                    #    "Architectural Sketch",
                    #    "Beautiful Outdoor Background",
                    #    "Fashion Cloth", 
                    #    "Pencil Sketch",
                    #    "Slate Pencil Mix"]
IMAGE_SIZE_CHOICES = [512]
                         
FREE_MEMORY_THRESHOLD = 7864320000 # 8864320000 # 7864320000 # 10050223473 # bytes 7340032000
FREE_MEMORY_PERCENTAGE_THRESHOLD = 0.39

INITIAL_SAMPLING_STEPS = 25
INITIAL_CFG = 7
INITIAL_SEED = 246372
INITIAL_IMAGE_SIZE = 512

NUM_USER_PER_GPU = 3

import torch

NUM_GPU = torch.cuda.device_count()


def get_concurrency_limit():
    import os
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    total_free = 0
    if cuda_visible_devices:
        cuda_ids = cuda_visible_devices.split(',')
        for device_id, cuda_id in enumerate(cuda_ids):
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(int(cuda_id))
            info = nvmlDeviceGetMemoryInfo(h)
            total_free += info.total 
    else:
        for i in range(torch.cuda.device_count()):
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(int(cuda_id))
            info = nvmlDeviceGetMemoryInfo(h)
            total_free += info.total 
    return total_free // FREE_MEMORY_THRESHOLD
            
CONCURRENCY_LIMIT = get_concurrency_limit()


