import warnings, os, gc, time, json, csv, copy, random
import logging as logger

import torch
import gradio as gr
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import logging
from transformers import CLIPImageProcessor
from datetime import datetime
from pynvml import *

import config
from config import MODEL_NAME_PATH_MAP, CRED_FPATH, SAFETY_CHECK, \
                   FREE_MEMORY_THRESHOLD, NUM_USER_PER_GPU, IMAGE_STYLE_CHOICES

import nltk
from nltk.tokenize import word_tokenize

# Ensure that the necessary NLTK data files are downloaded
nltk.download('punkt')

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# logger.basicConfig(filename=os.path.join("user-data","sys_info.log"),
#                     filemode='a',
#                     format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
#                     datefmt='%Y-%m-%d-%H:%M:%S',
#                     level=os.environ.get("LOGLEVEL", "INFO"))
logger.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
MYLOGGER = logger.getLogger()


def byte2mb(bytes_value):
    # 1 Megabyte (MB) is 1,048,576 Bytes (1024 * 1024)
    megabytes_value = bytes_value / (1024 * 1024)
    return megabytes_value


class GPUMonitor:
    def __init__(self):
        print('GPU monitor created ...')
        self.allocation = {} # {'device id': {using, cuda_id}}
        
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

        if cuda_visible_devices:
            cuda_ids = cuda_visible_devices.split(',')
            for i, cuda_id in enumerate(cuda_ids):
                self.allocation[int(i)] = {
                    "cuda_id": int(cuda_id)
                }
        else:
            for i in range(torch.cuda.device_count()): 
                self.allocation[int(i)] = {
                    "cuda_id": int(i)
                }
    
    def get_device_id(self, username):
        all_cuda_remain_avg = 0
        
        devices = list(self.allocation.items())
        random.shuffle(devices)  # Randomly shuffle the devices
        
        for device_id, info in devices:
        # for device_id, info in self.allocation.items():
            remain_mem = self._get_freememory(info['cuda_id'])
            if remain_mem <= FREE_MEMORY_THRESHOLD: 
                all_cuda_remain_avg += remain_mem
                continue
            
            cuda_id = info['cuda_id']
            MYLOGGER.info(f"---- GPU Monitor gives USER {username} cuda {cuda_id}, "\
                            f"device {device_id}, remaining : {byte2mb(remain_mem)} MB")
            return device_id, info['cuda_id'], remain_mem
        
        all_cuda_remain_avg /= len(self.allocation.keys())
        return None, None, byte2mb(all_cuda_remain_avg)
        
    def _get_freememory(self, cuda_id):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(cuda_id)
        info = nvmlDeviceGetMemoryInfo(h)
        return info.free # bytes
        

GPU_monitor = GPUMonitor() 

################################################################################

def count_tokens(input_string):
    tokens = word_tokenize(input_string)
    return len(tokens)

def debug_fn(user_data, loaded_model):
    gr.Warning("DEBUGGING ...")
    print("\n================================")
    print('user_data: ', user_data)
    print("loaded_model: ")
    print(loaded_model)
    for device_id, info in GPU_monitor.allocation.items(): 
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(info['cuda_id'])
        info = nvmlDeviceGetMemoryInfo(h)
        print(f"cuda {device_id} remain: {byte2mb(info.free)} MB")
    print("================================\n")

def get_auth_cred(username, password):
    """
    Return True if username and password match in the credential file.
    """
    with open(CRED_FPATH, encoding='utf-8') as f:
        cred = json.load(f)
    if username not in cred or cred[username] != password:
        return False
    return True

def create_greeting(user_data, request: gr.Request):
    user_data['username'] = request.username
    welcome_msg = f"Welcome to Text2Image Generation,  {user_data['username']}!"
    MYLOGGER.info(f">>>>>>>> USER {user_data['username']} Login.")
    final_return = [
        user_data,
        welcome_msg, 
        gr.update(visible=True), # image_style
        gr.update(visible=True), # image_size
        gr.update(visible=True), # prompt
        gr.update(visible=True), # negative_prompt
        gr.update(visible=True), # sampling_steps
        gr.update(visible=True), # cfg_scale
        gr.update(visible=True), # seed
        gr.update(visible=True), # generate_button
        gr.update(visible=True), # logout_button
        gr.update(visible=True), # image_output
        gr.update(visible=True), # info_output
    ]
    return final_return

def satisfaction_slider_change(satisfaction):
    if int(satisfaction) == 0:
        # gr.Warning("Please choose your satisfaction.")
        return gr.update(interactive=False) # save btn
    else:
        return gr.update(interactive=True) # save btn

def handle_save(user_data, satisfaction, why_unsatisfied):
    """
    Returns:
        user_data: gr.State({})
        satisfaction: gr.Slider()
        why_unsatisfied: gr.Textbox()
        save_button: gr.Button()
        generate_button: gr.Button() 
    """
    user_data['satisfaction'] = satisfaction
    user_data['why_unsatisfied'] = why_unsatisfied
    
    #********** save to csv ***************************************************
    store_fpath = config.user_data_store_fpath
    # Fetch the last timestamp from the CSV file if it exists
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H:%M:%S}.{:02.0f}'.format(cur_time, cur_time.microsecond / 10000.0)
    last_timestamp = None
    if os.path.exists(store_fpath):
        with open(store_fpath, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] != user_data['username']:
                    continue
                last_timestamp = row[2]  # Assuming the third column is the timestamp

    # Determine the new timestamp
    if last_timestamp is None:
        last_timestamp = 0
    else:
        last_timestamp = int(last_timestamp) + 1

    user_data['timestamp'] = last_timestamp

    # Prepare data in the specified format
    row = [
        cur_time,
        user_data['username'],
        last_timestamp,
        user_data['image_style'],
        user_data['image_size'],
        user_data['prompt'],
        user_data['negative_prompt'],
        user_data['sampling_steps'],
        user_data['cfg_scale'],
        user_data['seed'],
        user_data['satisfaction'],
        user_data['why_unsatisfied'],
    ]

    with open(store_fpath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    # gr.Info("Successfully saved info. If you want to save the image, "\
    #         "please click the button on the top right of image")
    #*************************************************************
    info_output = copy.deepcopy(user_data)
    info_output['saved_time'] = cur_time
    info_output.pop('timestamp', None)
    
    user_data["satisfaction"] = 0
    user_data['why_unsatisfied'] = ""
    
    final_return = [
        user_data, 
        gr.update(visible=False, value=0, interactive=True), # satisfaction
        gr.update(visible=False, value=""), # why_unsatisfied
        gr.update(visible=False, interactive=False), # save_button
        gr.update(visible=True, interactive=True), # generate_button
    ]
    for _ in range(7):
        final_return.append(gr.update(interactive=True))
    final_return.append(info_output)
    return final_return
        
def helper_init_model(image_style, image_size):
    safetensor_path = MODEL_NAME_PATH_MAP[image_style]
    
    safety_checker = StableDiffusionSafetyChecker\
                        .from_pretrained("CompVis/stable-diffusion-safety-checker")
    
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    loaded_model = StableDiffusionPipeline.from_single_file(safetensor_path,
                                                            extract_ema=True,
                                                            safety_checker=safety_checker,
                                                            feature_extractor=feature_extractor,
                                                            image_size=image_size)
    return loaded_model

def update_input(user_data, image_style, image_size, prompt, negative_prompt, 
                 sampling_steps, cfg_scale, seed):
    user_data['image_style'] = image_style 
    user_data['image_size'] = image_size 
    user_data['prompt'] = prompt 
    user_data['negative_prompt'] = negative_prompt 
    user_data['sampling_steps'] = sampling_steps 
    user_data['cfg_scale'] = cfg_scale 
    user_data['seed'] = seed 
    return user_data

def disable_component(*argv):
    final_res = []
    for arg in argv:
        final_res.append(gr.update(interactive=False))
    return final_res

def enable_component(*argv):
    final_res = []
    for arg in argv:
        final_res.append(gr.update(interactive=True))
    return final_res

def offload_model(username, loaded_model, device_id):
    if loaded_model is None:
        return
    try:
        loaded_model.to('cpu')
        gc.collect()
        del loaded_model.feature_extractor, \
            loaded_model.image_encoder, \
            loaded_model.safety_checker, \
            loaded_model.scheduler, \
            loaded_model.text_encoder, \
            loaded_model.tokenizer, \
            loaded_model.unet, \
            loaded_model.vae
        torch.cuda.empty_cache()
        del loaded_model
        gc.collect()
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()   
            gc.collect()
        loaded_model = None
    except Exception as e:
        MYLOGGER.info(f"ERROR message from USER {username} ----------------- offload model")
        MYLOGGER.info(str(e))

def process_error_in_generation(username, e, loaded_model, device_id, desc=None):
    MYLOGGER.info(f"ERROR message from USER {username} ------------- {desc}")
    
    MYLOGGER.info(str(e))
    if 'out of memory' in str(e):
        gr.Warning("GPU is busy. Please regenerate the image after we finish cleaning GPU memory.")
    else:
        gr.Warning("Please contact the staff. Something wrong with the server.")
    if loaded_model is None:
        return
    offload_model(username, loaded_model, device_id)
    
def preallocate_memory(cuda_id, num_bytes):
    # Determine the number of elements needed based on the size of float (4 bytes per float)
    num_elements = num_bytes // 4  # Using float32 (4 bytes per element)
    
    # Calculate the shape of the tensor
    # Assuming a 1D tensor for simplicity
    tensor_shape = (num_elements,)
    
    with torch.cuda.device(cuda_id):
        dummy_tensor = torch.cuda.FloatTensor(*tensor_shape)
        dummy_tensor.fill_(0)
    
def handle_generation(user_data):
    
    early_return = [] # 14 in total
    for i in range(5): # post_ui
        early_return.append(gr.update())
    for i in range(7): # group ui
        early_return.append(gr.update(visible=True, interactive=True)) 
    early_return.append(gr.update(visible=True, interactive=True)) # generate btn
    early_return.append(gr.update(visible=True, interactive=True)) # logout btn
    
    if user_data['image_style'] is None or user_data['image_style'] not in IMAGE_STYLE_CHOICES:
        gr.Warning("Please select your imaging style!")
        return early_return
    
    if user_data['prompt'] == '' and user_data['negative_prompt'] == '' or \
       user_data['prompt'].isspace() and user_data['negative_prompt'].isspace():
        gr.Warning("Please provide some prompt!")
        return early_return
    
    try:
        int_value = int(user_data['sampling_steps'])
    except:
        gr.Warning("Please provide numeric value for sampling steps!")
        return early_return
    
    try:
        int_value = int(user_data['cfg_scale'])
    except:
        gr.Warning("Please provide numeric value for cfg_scale!")
        return early_return
    
    try:
        int_value = int(user_data['seed'])
    except:
        gr.Warning("Please provide numeric value for seed!")
        return early_return
    
    
    MYLOGGER.info(f"---- USER {user_data['username']}: ---- handle generation")
    
    username = user_data['username']
    
    # Allocate gpu
    gr.Info("Loading the image style ... ")
    loaded_model = helper_init_model(user_data['image_style'], user_data['image_size'])
    MYLOGGER.info(f"---- USER {user_data['username']}: finished cpu model")
    
    trigger_warning = True
    device_id, cuda_id, cuda_usage = GPU_monitor.get_device_id(username)
    
    gr.Info("Waiting for idle GPU ... ")
    while device_id is None:
        if trigger_warning:
            trigger_warning = False
            MYLOGGER.info(f"---- USER {username}: ---- waiting for GPU. Avg mem remains {cuda_usage} MB")
        device_id, cuda_id, cuda_usage = GPU_monitor.get_device_id(username)
    
    # # preallocate mem
    # try:
    #     MYLOGGER.info(f"---- USER {username}: try to preallocate "\
    #                   f"{byte2mb(FREE_MEMORY_THRESHOLD)} MB on cuda {cuda_id}")
    #     preallocate_memory(device_id, FREE_MEMORY_THRESHOLD)
    # except Exception as e:
    #     process_error_in_generation(username, e, loaded_model, device_id, desc="preallocate error")
    #     loaded_model = None
    #     return early_return
        
    # Load model to GPU
    try:
        gr.Info("Found GPU. Trying to load model to it ... ")
        loaded_model.to(f"cuda:{device_id}")
    except Exception as e:
        process_error_in_generation(username, e, loaded_model, device_id, desc="load model to gpu")
        loaded_model = None
        return early_return
    
    # Generate image
    try:
        image_res = loaded_model(
                        prompt=user_data['prompt'],
                        negative_prompt=user_data['negative_prompt'], 
                        num_inference_steps=user_data['sampling_steps'], # sampling steps
                        guidance_scale=user_data['cfg_scale'], # cfg
                        generator=torch.manual_seed(user_data['seed']),
                    )
        
        nsfw_detected = image_res['nsfw_content_detected'][0]
        
        image = image_res.images[0] # PIL.Image.Image
        
    except Exception as e:
        process_error_in_generation(username, e, loaded_model, device_id, desc="image generation")
        loaded_model = None
        return early_return
    
    offload_model(username, loaded_model, device_id)
    loaded_model = None
    
    ############################################################################
    
    if SAFETY_CHECK:
        if nsfw_detected:
            gr.Warning("Please use proper prompt!")
            MYLOGGER.info(f"---- USER {username} : ---- improper prompt")
            return early_return
    
    MYLOGGER.info(f"---- USER {username} successfully generated image")
    
    info_output = copy.deepcopy(user_data)
    info_output.pop('timestamp', None)
    
    final_return = [image, info_output]
    final_return.append(gr.update(visible=True, interactive=False)) # save button
    final_return.append(gr.update(visible=True, interactive=True)) # satisfaction
    final_return.append(gr.update(visible=True, interactive=True)) # why unsatisfied
    for i in range(7): # group ui
        final_return.append(gr.update(visible=True, interactive=False))
    final_return.append(gr.update(visible=True, interactive=False)) # generate btn
    final_return.append(gr.update(visible=True, interactive=True)) # logout btn
    
    return final_return

################################################################################

def get_cuda_info():
    print('################################################################')
    print("Number of available devices:", torch.cuda.device_count())
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    assert torch.cuda.device_count() > 0, \
           "\n************* No GPU available *************\n"
    
    if cuda_visible_devices:
        print(f"CUDA_VISIBLE_DEVICES id: {cuda_visible_devices}")
        cuda_ids = cuda_visible_devices.split(',')
        for i, cuda_id in enumerate(cuda_ids):
            print(f"    cuda ID {cuda_id}, device ID {i}: {torch.cuda.get_device_name(i)}")
    else:
        for i in range(torch.cuda.device_count()):
            print(f"    cuda ID {i}, device ID {i}: {torch.cuda.get_device_name(i)}")
    print('################################################################')
