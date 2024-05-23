import warnings, os, gc

import torch
import gradio as gr

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import logging
from transformers import CLIPImageProcessor


logging.set_verbosity_error()
warnings.filterwarnings("ignore")
MODEL_NAME_PATH_MAP = {
    "Oil Painting": "models/oilPaintingV10.safetensors",
    "Watercolor": "models/colorwater-v4.safetensors", 
    "MoXin (traditional Chinese Painting)": "models/MoXin-v1.safetensors",
}


def clear_cuda_memory(loaded_model):
    print("ready to clear cuda memory")
    if loaded_model is None: return
    
    device = loaded_model.device   
    print('model was on device: ', device)
    
    gc.collect()
    with torch.no_grad(): 
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    del loaded_model
    
    print("GPU memory released")
    return gr.State(None)

def get_cuda_info():
    print('################################################################')
    print("Number of available devices:", torch.cuda.device_count())
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    assert torch.cuda.device_count() > 0, \
           "\n************* No GPU available *************\n"
    
    if cuda_visible_devices:
        print(f"CUDA_VISIBLE_DEVICES id: {cuda_visible_devices}")
        device_ids = cuda_visible_devices.split(',')
        for i, device_id in enumerate(device_ids):
            print(f"    Device ID {device_id}: {torch.cuda.get_device_name(i)}")
    else:
        for i in range(torch.cuda.device_count()):
            print(f"    Device ID {i}: {torch.cuda.get_device_name(i)}")
    print('################################################################')
    
def find_most_idle_gpu():
    num_gpus = torch.cuda.device_count()
    max_free_memory = 0
    best_gpu = 0
    
    for i in range(num_gpus):
        free_memory = torch.cuda.get_device_properties(i).total_memory - \
                      torch.cuda.memory_allocated(i)
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i
    return best_gpu

def load_model(user_data, image_style, loaded_model, image_size):
    """
    Returns:
        usder_data: gr.State({})
        image_style: gr.Dropdown()
        loaded_model: gr.State()
    """
    
    if image_style is None and "image_style" not in user_data:
        return user_data, image_style, loaded_model
    
    if loaded_model is not None: clear_cuda_memory(loaded_model)
    
    user_data['image_style'] = image_style
    user_data['image_size'] = image_size
    
    safetensor_path = MODEL_NAME_PATH_MAP[image_style]
    
    # progress(0.2, "Loading safety checker...")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    
    # progress(0.4, "Loading feature extractor...")
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # progress(0.6, "Loading model...")
    loaded_model = StableDiffusionPipeline.from_single_file(safetensor_path,
                            extract_ema=True,
                            safety_checker=safety_checker,
                            feature_extractor=feature_extractor,
                            image_size=image_size)
    
    best_gpu = find_most_idle_gpu()
    device = torch.device(f'cuda:{best_gpu}')
    print("===== ", image_style)
    before = torch.cuda.get_device_properties(best_gpu).total_memory - torch.cuda.memory_allocated(best_gpu)
    before /= (1024 ** 2)
    #*********************
    loaded_model.to(device)
    
    after = torch.cuda.get_device_properties(best_gpu).total_memory - torch.cuda.memory_allocated(best_gpu)
    after /= (1024 ** 2)
    print(f"model usage of cuda: {(before - after):2f} MB")
    
    

    # progress(1.0, "Finish model loading.")
    # gr.Info("Finish model loading.")
    print(f"model loaded on device {best_gpu}...")
    return user_data, image_style, loaded_model

def change_image_size(user_data, image_size, loaded_model):
    """Change image size for generation. Reload model if necessary.
    
    Returns:
        user_data: gr.State({})
        image_size: gr.Dropdown()
        loaded_model: gr.State()
    """
    if "image_style" not in user_data:
        user_data['image_size'] = image_size
        return user_data, image_size, loaded_model
    
    if image_size not in user_data:
        user_data['image_size'] == image_size
        
    if user_data['image_size'] == image_size and loaded_model is not None:
        return user_data, image_size, loaded_model
    
    clear_cuda_memory(loaded_model)
    
    user_data['image_size'] = image_size
    user_data, _, loaded_model, = load_model(user_data, user_data['image_style'], 
                                             loaded_model, image_size)
    return user_data, image_size, loaded_model

def generate_image(loaded_model, model_input):
    """Generate images from loaded model based on user input

    Args:
        loaded_model: gr.State(StableDiffusionPipeline)
        model_input: { =====> user_data
            "image_style": image_style,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampling_steps": sampling_steps,
            "cfg_scale": cfg_scale,
            "seed": seed
        }

    Returns:
        image: gr.Image()
        info: gr.Json()
    """
    
    print(f"Generated Image with model: {model_input['image_style']}")
    info = {
        "username": model_input['username'],
        "image_style": model_input['image_style'],
        "image_size": model_input['image_size'],
        "prompt": model_input['prompt'],
        "negative_prompt": model_input['negative_prompt'],
        "sampling_steps": model_input['sampling_steps'],
        "cfg_scale": model_input['cfg_scale'],
        "seed": model_input['seed']
    }

    image = loaded_model(
        prompt=model_input['prompt'],
        negative_prompt=model_input['negative_prompt'], 
        num_inference_steps=model_input['sampling_steps'], # sampling steps
        guidance_scale=model_input['cfg_scale'], # cfg
        generator=torch.manual_seed(model_input['seed']),
    )
    
    image = image.images[0] # PIL.Image.Image
    
    return image, info




