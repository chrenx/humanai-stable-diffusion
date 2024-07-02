from pynvml import *

AUTH_MSG_FPATH = "data_agreement.html" # data agreement file path
USER_DATA_STORE_FPATH = "user-data/public_user_data.csv" 

SAFETY_CHECK = True

CRED_FPATH = "authentication.json"

MODEL_NAME_PATH_MAP = {
    "Oil Painting": "models/oilPaintingV10.safetensors",
    "Watercolor": "models/colorwater-v4.safetensors", 
    "MoXin (traditional Chinese Painting)": "models/MoXin-v1.safetensors",
    "Outdoor Background": "models/background_beautiful_outdoor.safetensors",
    "Exterior Design": "models/exterior_design.safetensors",
    "Fashion Cloth": "models/fashion_cloth.safetensors",
    "Fashion Sketch": "models/fashion_sketch.safetensors",
    "Game Wallpaper": "models/game_wallpaper.safetensors",
    "Interior Design": "models/interior_design.safetensors",
    "Pencil Sketch": "models/pencil_sketch.safetensors",
    "Architecture Sketch": "models/sketch_achitectural_design.safetensors",
    "Car Sketch": "models/sketch_car.safetensors",
    "Slate Pencil": "models/slate_pencil_mix.safetensors",
}

# IMAGE_STYLE_CHOICES = ["Oil Painting"]
IMAGE_STYLE_CHOICES = ["Oil Painting", 
                       "Watercolor", 
                       "MoXin (traditional Chinese Painting)",     
                       "Outdoor Background",
                       "Exterior Design",
                       "Fashion Cloth",
                       "Fashion Sketch",
                       "Game Wallpaper",
                       "Interior Design",
                       "Pencil Sketch",
                       "Architecture Sketch",
                       "Car Sketch",
                       "Slate Pencil"]

IMAGE_SIZE_CHOICES = [
    "384x512 (4:3 ratio)", "300x400 (4:3 ratio)", "240x320 (4:3 ratio)",
    "288x512 (16:9 ratio)", "225x400 (16:9 ratio)", "180x320 (16:9 ratio)",
    "512x512 (1:1 ratio)", "400x400 (1:1 ratio)", "320x320 (1:1 ratio)"
]
                         
FREE_MEMORY_THRESHOLD = 9178892800 # 8178892800 # 7864320000 # 8864320000 # 7864320000 # 10050223473 # bytes 7340032000
FREE_MEMORY_PERCENTAGE_THRESHOLD = 0.39

INITIAL_SAMPLING_STEPS = 25
INITIAL_CFG = 7
INITIAL_SEED = 246372
INITIAL_IMAGE_SIZE = 512
MAX_NUM_IMAGES = 4
LIMITED_TOKEN = 77
NUM_LISTEN = 2


import torch

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
            h = nvmlDeviceGetHandleByIndex(int(i))
            info = nvmlDeviceGetMemoryInfo(h)
            total_free += info.total 
    return total_free // FREE_MEMORY_THRESHOLD
 
# CONCURRENCY_LIMIT = get_concurrency_limit()
CONCURRENCY_LIMIT = 3

# from diffusers import StableDiffusionPipeline
# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
# from transformers import CLIPImageProcessor
# from tqdm import tqdm

# def preload_all_model():
#     preload_model = {}
#     device_ids = []
#     cuda_ids = []
#     cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
#     if cuda_visible_devices:
#         print("find visible cuda")
#         cuda_ids = cuda_visible_devices.split(',')
#         print("all visible cudas")
#         print(cuda_ids)
#         for i, cuda_id in enumerate(cuda_visible_devices):
#             # print(f"dealing with device {i} cuda {cuda_id}")
#             device_ids.append(i)
#             cuda_ids.append(cuda_id)
#     else:
#         print("use all cuda")
#         for i in range(torch.cuda.device_count()): 
#             device_ids.append(i)
#             cuda_ids.append(i)
#     for style in tqdm(IMAGE_STYLE_CHOICES, total=len(IMAGE_STYLE_CHOICES), desc="Loading Model"):
#         safetensor_path = MODEL_NAME_PATH_MAP[style]
    
#         safety_checker = StableDiffusionSafetyChecker\
#                             .from_pretrained("CompVis/stable-diffusion-safety-checker")
        
#         feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
#         preload_model[style] = StableDiffusionPipeline.from_single_file(safetensor_path,
#                                                                 extract_ema=True,
#                                                                 safety_checker=safety_checker,
#                                                                 feature_extractor=feature_extractor,
#                                                                 image_size=INITIAL_IMAGE_SIZE)
#         gpu_not_found = True
#         for device_id, cuda_id in zip(device_ids, cuda_ids):
#             nvmlInit()
#             h = nvmlDeviceGetHandleByIndex(int(cuda_id))
#             info = nvmlDeviceGetMemoryInfo(h)
#             remain_mem = info.free
#             if remain_mem < FREE_MEMORY_THRESHOLD:
#                 continue            
#             gpu_not_found = False
#             preload_model[style] = preload_model[style].to(f"cuda:{device_id}")
#             break
#         if gpu_not_found:
#             raise f"not enough gpu memory left for loading {style}"
        
#     return preload_model

# PRELOAD_MODELS = preload_all_model()
