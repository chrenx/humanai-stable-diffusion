import torch

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from tqdm import tqdm

from config import IMAGE_STYLE_CHOICES, MODEL_NAME_PATH_MAP, INITIAL_IMAGE_SIZE, \
                   FREE_MEMORY_THRESHOLD

from pynvml import *

def preload_all_model():
    preload_model = {}
    device_ids = []
    cuda_ids = []
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        print("find visible cuda")
        cuda_ids = cuda_visible_devices.split(',')
        print("all visible cudas")
        print(cuda_ids)
        for i, cuda_id in enumerate(cuda_visible_devices):
            # print(f"dealing with device {i} cuda {cuda_id}")
            device_ids.append(i)
            cuda_ids.append(cuda_id)
    else:
        print("use all cuda")
        for i in range(torch.cuda.device_count()): 
            device_ids.append(i)
            cuda_ids.append(i)
    for style in tqdm(IMAGE_STYLE_CHOICES, total=len(IMAGE_STYLE_CHOICES), desc="Loading Model"):
        safetensor_path = MODEL_NAME_PATH_MAP[style]
    
        safety_checker = StableDiffusionSafetyChecker\
                            .from_pretrained("CompVis/stable-diffusion-safety-checker")
        
        feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        preload_model[style] = StableDiffusionPipeline.from_single_file(safetensor_path,
                                                                extract_ema=True,
                                                                safety_checker=safety_checker,
                                                                feature_extractor=feature_extractor,
                                                                image_size=INITIAL_IMAGE_SIZE)
        gpu_not_found = True
        for device_id, cuda_id in zip(device_ids, cuda_ids):
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(int(cuda_id))
            info = nvmlDeviceGetMemoryInfo(h)
            remain_mem = info.free
            if remain_mem < FREE_MEMORY_THRESHOLD:
                continue            
            gpu_not_found = False
            print(f"\n-------- load {style} to cuda: {cuda_id}\n")
            preload_model[style] = preload_model[style].to(f"cuda:{device_id}")
            break
        if gpu_not_found:
            raise f"not enough gpu memory left for loading {style}"
        
    return preload_model

