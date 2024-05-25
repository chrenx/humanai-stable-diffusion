
auth_cred_fpath = "auth.json" # user credentials
auth_message_fpath = "data_agreement.html" # data agreement file path
user_data_store_fpath = "user-data/public_user_data.csv" 

SAFETY_CHECK = True

CRED_FPATH = "auth.json"
MODEL_NAME_PATH_MAP = {
    "Oil Painting": "models/oilPaintingV10.safetensors",
    "Watercolor": "models/colorwater-v4.safetensors", 
    "MoXin (traditional Chinese Painting)": "models/MoXin-v1.safetensors",
}

IMAGE_STYLE_CHOICES = ["Oil Painting", "Watercolor", "MoXin (traditional Chinese Painting)"]
IMAGE_SIZE_CHOICES = [512]

FREE_MEMORY_THRESHOLD = 0.75e+10

INITIAL_SAMPLING_STEPS = 25
INITIAL_CFG = 7
INITIAL_SEED = 246372
INITIAL_IMAGE_SIZE = 512

NUM_USER_PER_GPU = 1

import torch

NUM_GPU = torch.cuda.device_count()




