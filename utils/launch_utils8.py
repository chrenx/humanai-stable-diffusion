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

from config import MODEL_NAME_PATH_MAP, CRED_FPATH, SAFETY_CHECK, \
                   IMAGE_STYLE_CHOICES, INITIAL_IMAGE_SIZE, FREE_MEMORY_THRESHOLD, \
                   USER_DATA_STORE_FPATH, PRELOAD_MODELS

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

################################################################################

def count_tokens(input_string):
    tokens = word_tokenize(input_string)
    return len(tokens)

def debug_fn(user_data):
    gr.Warning("DEBUGGING ...")
    print("\n================================")
    print('user_data: ', user_data)
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
    store_fpath = USER_DATA_STORE_FPATH
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
    ]
    
    for _ in IMAGE_STYLE_CHOICES:
        final_return.append(gr.update(visible=True, interactive=True))
    
    for _ in range(6): # group ui
        final_return.append(gr.update(visible=True, interactive=True))
    final_return.append(info_output)
    
    return final_return
        
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
    # final_res.append(gr.update(visible=False)) # input col
    return final_res

def handle_generation(user_data):
    
    #* Check early return ******************************************************
    early_return = [] # 13 in total
    for i in range(5): # post_ui
        early_return.append(gr.update())
    for i in range(6): # group ui
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
    #***************************************************************************
    
    
    MYLOGGER.info(f"---- USER {user_data['username']}: ---- handle generation")
    
    username = user_data['username']
    
    # Generate image
    try:
        image_res = PRELOAD_MODELS[user_data['image_style']](
                        prompt=user_data['prompt'],
                        negative_prompt=user_data['negative_prompt'], 
                        num_inference_steps=user_data['sampling_steps'], # sampling steps
                        guidance_scale=user_data['cfg_scale'], # cfg
                        generator=torch.manual_seed(user_data['seed']),
                        num_images_per_prompt=4,
                    )
        
        nsfw_detected = image_res['nsfw_content_detected'][0]
        
        image = image_res.images[0] # PIL.Image.Image
        
        torch.cuda.empty_cache()
        
    except Exception as e:
        MYLOGGER.info(f"ERROR message from USER {username} ------------- generate image")
        MYLOGGER.info(str(e))
        return early_return
    
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
    for i in range(6): # group ui
        final_return.append(gr.update(visible=True, interactive=False))
    for _ in IMAGE_STYLE_CHOICES:
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
