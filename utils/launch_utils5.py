import warnings, os, gc, time, json, csv, copy
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
                   FREE_MEMORY_PERCENTAGE_THRESHOLD, NUM_USER_PER_GPU

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
        cuda_remain = 0
        for device_id, info in self.allocation.items():
            
            free_memory_percentage = self._get_freememory_percentage(info['cuda_id'])
            if free_memory_percentage <= FREE_MEMORY_PERCENTAGE_THRESHOLD: # 0.3
                cuda_remain += free_memory_percentage
                continue
            
            cuda_id = info['cuda_id']
            MYLOGGER.info(f"---- GPU Monitor gives USER {username} cuda {cuda_id}, "\
                            f"device {device_id}, remaining percentage: {free_memory_percentage}")
            return device_id, cuda_remain
        cuda_remain /= len(self.allocation.keys())
        return None, cuda_remain
        
    def _get_freememory_percentage(self, cuda_id):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(cuda_id)
        info = nvmlDeviceGetMemoryInfo(h)
        remain_percentage = info.free / info.total 
        return remain_percentage
        

GPU_monitor = GPUMonitor() 

################################################################################

def count_tokens(input_string):
    tokens = word_tokenize(input_string)
    return len(tokens)

def debug_fn(user_data, loaded_model):
    print("\n================================")
    print('user_data: ', user_data)
    print("loaded_model: ")
    print(loaded_model)
    for device_id, info in GPU_monitor.allocation.items(): 
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(info['cuda_id'])
        info = nvmlDeviceGetMemoryInfo(h)
        remain_percentage = info.free / info.total
        print(f"cuda {device_id} remain: ", remain_percentage * 100, "%")
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

def update_image_size(user_data, image_size):
    """Change image size for generation. Reload model if necessary.
    Returns:
        user_data: gr.State({})
        image_size: gr.Dropdown()
    """
    user_data['image_size'] = image_size
    return user_data, image_size

def update_image_style(user_data, image_style):
    """
    Returns:
        user_data: gr.State({})
        image_style: gr.Dropdown()
    """
    user_data['image_style'] = image_style
    return user_data, image_style


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
        
def handle_logout(user_data, demo):
    """Clear everything after logout
    Returns:
        logined_username: gr.State()
        user_data: gr.State({})
        loaded_model: gr.State()
        image_style: gr.Dropdown()
    """
    print('........handling logout')
    demo.unload(lambda user_data: MYLOGGER(f"USER LOGOUT: {user_data['username']}, web unload"))
    return 
    # return response


def post_gen_ui(loaded_model):
    if loaded_model is not None:
        collected = gc.collect()
    
        print("collected before: ", collected)
        loaded_model.to('cpu')
        collected = gc.collect()
        print("collected mid: ", collected)
        torch.cuda.empty_cache()   
        collected = gc.collect()
        print("collected after: ", collected)
        
        torch.cuda.empty_cache()
        loaded_model.to('cpu')
    return loaded_model

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

def disable_generate():
    return gr.update(interactive=False)
    
def handle_init_model(user_data, image_style, image_size):
    user_data['image_style'] = image_style
    user_data['image_size'] = image_size
    return gr.update(), gr.update(interactive=True)

def pre_gen_ui(user_data, image_style, image_size, prompt, negative_prompt, 
                sampling_steps, cfg_scale, seed, loaded_model):
    
    # print("这里1", request.session_hash)

    user_data['image_style'] = image_style 
    user_data['image_size'] = image_size 
    user_data['prompt'] = prompt 
    user_data['negative_prompt'] = negative_prompt 
    user_data['sampling_steps'] = sampling_steps 
    user_data['cfg_scale'] = cfg_scale 
    user_data['seed'] = seed 
    
    early_return = [user_data]
    for i in range(8):
        early_return.append(gr.update())
    early_return.append(True) # gen_stop_flag
    early_return.append(None) # loaded_model

    if user_data['image_style'] is None:
        gr.Warning("Please select your imaging style!")
        return early_return
    
    if user_data['prompt'] == '' and user_data['negative_prompt'] == '' or \
       user_data['prompt'].isspace() and user_data['negative_prompt'].isspace():
        gr.Warning("Please provide some prompt!")
        return early_return
    
    num_tokens = count_tokens(user_data['prompt'])
    if count_tokens(user_data['prompt']) > 77:
        gr.Warning(f"Prompt is too long, Currently {num_tokens} tokens.")
        return early_return
    
    final_return = [user_data]
    for i in range(7):
        final_return.append(gr.update(interactive=False))
    final_return.append(gr.update(interactive=False)) # generation button
    final_return.append(False) # gen_stop_flag
    
    # load model to gpu:
    loaded_model = helper_init_model(user_data['image_style'], user_data['image_size'])
    
    username = user_data['username']
    
    device_id, cuda_usage = GPU_monitor.get_device_id(username)
    cuda_usage *= 100
    trigger_warning = True
    while device_id is None:
        if trigger_warning:
            trigger_warning = False
            MYLOGGER.info(f"---- USER {username}: ---- waiting for GPU")
            gr.Warning(f"Waiting for idle GPU. {cuda_usage:.2f}% of CUDA memory remains "\
                       "on server. Please don't leave.")
        device_id, cuda_usage = GPU_monitor.get_device_id(username)
        
    try:
        loaded_model.to(f"cuda:{device_id}")
    except Exception as e:
        print("Error --------")
        print(e)
        gr.Warning(f"GPU memory is not enough. Please regenerate.")
        del loaded_model
        torch.cuda.empty_cache()
        return early_return
    
    final_return.append(loaded_model)
    
    # print("这里2", request.session_hash)
    
    return final_return


def handle_generation(user_data, loaded_model, gen_stop_flag):
    
    early_return = [user_data]
    for i in range(6):
        early_return.append(gr.update())
    for i in range(7):
        early_return.append(gr.update(visible=True, interactive=True)) # group ui
    
    if gen_stop_flag:
        early_return.append(loaded_model)
        return early_return
    
    MYLOGGER.info(f"---- USER {user_data['username']}: ---- handle generation")
    
    username = user_data['username']
    
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
        print("ERROR message -----------------------------------------")
        print(e)
        print(f"USER {username} GPU error here...")
        print(loaded_model)
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
        torch.cuda.empty_cache()   
        gc.collect()
        loaded_model = None
        early_return.append(loaded_model)
        gr.Warning("Please regenerate the image.")
        return early_return
    
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
    torch.cuda.empty_cache()
    loaded_model = None
    
    ############################################################################
    
    if SAFETY_CHECK:
        if nsfw_detected:
            gr.Warning("Please use proper prompt!")
            MYLOGGER.info(f"---- USER {username} : ---- improper prompt")
            early_return[3] = gr.update(visible=True, interactive=True) # generate button
            early_return.append(loaded_model)
            return early_return
    
    info_output = copy.deepcopy(user_data)
    info_output.pop('timestamp', None)
    
    final_return = [user_data, image, info_output, gr.update(visible=False)]
    
    MYLOGGER.info(f"---- USER {username} successfully generated image")
    
    final_return.append(gr.update(visible=True, interactive=False)) # save button
    final_return.append(gr.update(visible=True, interactive=True)) # satisfaction
    final_return.append(gr.update(visible=True, interactive=True)) # why unsatisfied
    
    for i in range(7): # group ui
        final_return.append(gr.update(visible=True, interactive=False))
    
    final_return.append(loaded_model)
    return final_return

################################################################################

def clear_cuda_memory(loaded_model):
    if loaded_model is None: return
    
    device = loaded_model.device 
    
    collected = gc.collect()
    print("collected before: ", collected)
    del loaded_model
    collected = gc.collect()
    print("collected mid: ", collected)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()   
    collected = gc.collect()
    print("collected after: ", collected)
    
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
