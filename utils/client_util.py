import warnings, os, gc, time, json, csv, copy, random, socket, pickle, io, re
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
from PIL import Image

from config import CRED_FPATH, MAX_NUM_IMAGES, IMAGE_STYLE_CHOICES, USER_DATA_STORE_FPATH

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
        gr.update(visible=True), # page_1
        gr.update(visible=False), # page_2
        gr.update(visible=False), # page_2_2
        gr.update(visible=False) # page_2_3
    ]
    return final_return

def satisfaction_slider_change(user_data, *argv):
    # MAX_NUM_IMAGES arg in argv
    enable_save_btn = True
    for i in range(len(argv)):
        try:
            tmp = int(argv[i])
        except Exception as e:
            gr.Warning("Please rate the image properly.")
            MYLOGGER.info(f"------USER {user_data['username']} error: satisfaction_slider_change")
            MYLOGGER.info(e)
            return gr.update(interactive=False)
        if user_data['proper_save'][i] and int(argv[i]) == 0:
            enable_save_btn = False         
    return gr.update(interactive=enable_save_btn) # save btn

def handle_save(user_data, satisfaction_0, satisfaction_1, satisfaction_2, satisfaction_3,
                comment_0, comment_1, comment_2, comment_3):
    """
    Returns:
        user_data: gr.State({})
        satisfaction: gr.Slider()
        why_unsatisfied: gr.Textbox()
        save_button: gr.Button()
        generate_button: gr.Button() 
    """
    user_data['satisfaction_0'] = satisfaction_0
    user_data['satisfaction_1'] = satisfaction_1
    user_data['satisfaction_2'] = satisfaction_2
    user_data['satisfaction_3'] = satisfaction_3
    user_data['comment_0'] = comment_0
    user_data['comment_1'] = comment_1
    user_data['comment_2'] = comment_2
    user_data['comment_3'] = comment_3
    
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
        user_data['satisfaction_0'],
        user_data['satisfaction_1'],
        user_data['satisfaction_2'],
        user_data['satisfaction_3'],
        user_data['comment_0'],
        user_data['comment_1'],
        user_data['comment_2'],
        user_data['comment_3']
    ]
    
    headers = ['timestamp', 'username', 'last_timestamp', 'image_style', 'image_size', 'prompt',
               'negative_prompt', 'sampling_steps', 'cfg_scale', 'seed', 'satisfaction_0',
               'satisfaction_1', 'satisfaction_2', 'satisfaction_3', 'comment_0', 'comment_1',
               'comment_2', 'comment_3']

    file_exists = os.path.isfile(store_fpath)
    
    with open(store_fpath, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(store_fpath) == 0:
            writer.writerow(headers) # write headers
        writer.writerow(row)
    gr.Info("Successfully saved info. If you want to save the image, "\
            "please click the button on the top right of image")
    #*************************************************************
    info_output = copy.deepcopy(user_data)
    info_output['saved_time'] = cur_time
    info_output.pop('timestamp', None)
    
    
    final_return = [
        user_data, 
        info_output,
        gr.update(visible=True, interactive=False), # save_button
        gr.update(visible=True, interactive=True), # generate_next
    ]
    
    return final_return

def go_to_page_2():
    final_return = [
        gr.update(visible=False), # page_1
        gr.update(visible=True), # page_2
        gr.update(visible=True), # page_2_2
        gr.update(visible=True), # page_2_3
    ]
    for i in range(MAX_NUM_IMAGES): # all_col_image
        final_return.append(gr.update(value=None, visible=True))
    return final_return

def go_to_page_1():
    final_return = [
        gr.update(visible=True), # page_1
        gr.update(visible=False), # page_2
        gr.update(visible=False), # page_2_2
        gr.update(visible=False), # page_2_3
        gr.update(visible=True, interactive=False), # save_button
        gr.update(visible=True, interactive=False) # generate next
    ]
    all_image_output = []
    all_satisfaction = []
    all_comment = []
    
    for _ in range(MAX_NUM_IMAGES):
        all_image_output.append(gr.update(value=None, visible=True))
        all_satisfaction.append(gr.update(value=0, visible=True))
        all_comment.append(gr.update(value=None, visible=True))
        
    final_return += all_image_output + all_satisfaction + all_comment
        
    return final_return
    
def update_input(user_data, image_style, num_images, image_size, prompt, negative_prompt, 
                 sampling_steps, cfg_scale, seed):
    user_data['image_style'] = image_style
    user_data['num_images'] = int(num_images)
    user_data['image_size'] = image_size 
    user_data['prompt'] = prompt 
    user_data['negative_prompt'] = negative_prompt 
    user_data['sampling_steps'] = int(sampling_steps) 
    user_data['cfg_scale'] = int(cfg_scale) 
    user_data['seed'] = int(seed) 
    
    match = re.search(r"(\d+)x(\d+)", image_size)
    width = int(match.group(1))
    height = int(match.group(2))
    user_data['image_width'] = width
    user_data['image_height'] = height
    
    return user_data

def handle_generation(user_data):
    
    #* Check early return ******************************************************
    early_return = [user_data, gr.update(visible=True), gr.update(visible=False)] # page 1, 2
    early_return += [gr.update(visible=False), gr.update(visible=False)] # page_2_2, page_2_3
    early_return.append(user_data)
    for i in range(MAX_NUM_IMAGES): # all_col_image
        early_return.append(gr.update(visible=True)) 
    for i in range(MAX_NUM_IMAGES): # all_image_output
        early_return.append(gr.update(visible=True, value=None))
    for i in range(MAX_NUM_IMAGES): # all_satisfaction
        early_return.append(gr.update(visible=False, value=0)) 
    for i in range(MAX_NUM_IMAGES): # all_comment
        early_return.append(gr.update(visible=False, value=None)) 

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
        # send request to server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 65432))
        data = pickle.dumps(user_data)
        client_socket.sendall(data)

        # Receive the data from the server
        received_data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            received_data += packet

        # Unpickle the received data
        res = pickle.loads(received_data)
        
        # Deserialize images
        for i in range(user_data['num_images']):
            img = Image.open(io.BytesIO(res[f'image{i}']))
            res[f'image{i}'] = img

        client_socket.close()
        
    except Exception as e:
        MYLOGGER.info(f"ERROR message from USER {username} ------------- generate image")
        MYLOGGER.info(str(e))
        return early_return
    
    ############################################################################
    MYLOGGER.info(f"---- USER {username} successfully generated image")
    
    all_col_image = []
    all_image_output = []
    all_satisfaction = []
    all_comment = [] 
    
    all_improper = True
    
    user_data['proper_save'] = [False] * MAX_NUM_IMAGES
    
    for i in range(MAX_NUM_IMAGES):
        if i < int(user_data['num_images']):
            all_col_image.append(gr.update(visible=True))
            if res[f'nsfw{i}']: # improper image
                all_image_output.append(gr.update(value=None, 
                                                  label="IMPROPER image generated by model!", 
                                                  show_label=True))
                all_satisfaction.append(gr.update(visible=False, value=0))
                all_comment.append(gr.update(visible=False, value=None)) 
            else:
                user_data['proper_save'][i] = True
                all_improper = False
                all_image_output.append(gr.update(value=res[f'image{i}']))
                all_satisfaction.append(gr.update(visible=True, value=0))
                all_comment.append(gr.update(visible=True, value=None)) 
        else:
            all_col_image.append(gr.update(visible=True))
            all_image_output.append(gr.update(visible=True, value=None))
            all_satisfaction.append(gr.update(visible=False, value=0))
            all_comment.append(gr.update(visible=False, value=None))
    
    info_output = copy.deepcopy(user_data)
    info_output.pop('timestamp', None)
    
    #                          page_1                    page_2
    final_return = [user_data, gr.update(visible=False), gr.update(visible=True), 
                    gr.update(visible=True), gr.update(visible=True), # page_2_2, page_2_3
                    info_output]
    final_return += all_col_image + all_image_output + all_satisfaction + all_comment
        
    
    if all_improper:
        gr.Warning("All generated images are impropered. Consider using proper prompt.")
        return early_return
    
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
