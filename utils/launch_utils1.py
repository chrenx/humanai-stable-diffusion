import warnings, os, gc, time, json, csv
import logging as logger

import torch
import gradio as gr
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import logging
from transformers import CLIPImageProcessor

import config
from config import MODEL_NAME_PATH_MAP, CRED_FPATH, SAFETY_CHECK

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

logger.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
MYLOGGER = logger.getLogger()


def debug_fn(user_data):
    print('user_data: ', user_data)

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

def handle_generate(user_data, prompt, negative_prompt, sampling_steps, cfg_scale, seed):
    """
    Returns:
        user_data: gr.State()
        image_output: gr.Image()
        satisfaction: gr.Slider()
        why_unsatisfied: gr.Textbox()
        save_button: gr.Button()
        info_output: gr.Json()
        generate_button: gr.Button()
    """
    
    user_data['prompt'] = prompt 
    user_data['negative_prompt'] = negative_prompt 
    user_data['sampling_steps'] = sampling_steps 
    user_data['cfg_scale'] = cfg_scale 
    user_data['seed'] = seed 
    
    early_return = [user_data]
    for i in range(6):
        early_return.append(gr.update())

    if user_data['image_style'] is None:
        gr.Warning("Please select your imaging style!")
        return early_return
    
    if user_data['prompt'] == '' and user_data['negative_prompt'] == '' or \
       user_data['prompt'].isspace() and user_data['negative_prompt'].isspace():
        gr.Warning("Please provide some prompt!")
        return early_return
    
    if user_data['prompt'].isspace(): user_data['prompt'] = ""
    if user_data['negative_prompt'].isspace(): user_data['negative_prompt'] = ""

    image, loaded_model, nsfw_detected = load_and_generate_image(user_data)
    
    if SAFETY_CHECK:
        if nsfw_detected:
            MYLOGGER.info(f"---- USER {user_data['username']} improper prompt")
            gr.Warning("Please use proper prompt.")
            return early_return
    
    if image is None:
        gr.Warning("Something wrong with the image generation...")
        return early_return

    final_return = [
        user_data, 
        image,
        gr.update(visible=True), # satisfaction
        gr.update(visible=True), # why_unsatisfied
        gr.update(visible=True), # save_button
        user_data, # info
        gr.update(visible=False), # generate_button
    ]
    
    clear_cuda_memory(loaded_model, user_data)
    
    return final_return

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
    
    # if user_data['satisfaction'] is None:
    #     gr.Warning("Please choose your satisfaction with the image.")
    #     return [user_data, 
    #             gr.update(),  # satisfaction
    #             gr.update(),  # why_unsatisfied
    #             gr.update(),  # save_button
    #             gr.update()  # generate_button
    #     ]
    
    
    
    if (user_data['why_unsatisfied'].isspace() or user_data['why_unsatisfied'] == "") \
       and user_data['satisfaction'] < 5: 
           
        user_data['why_unsatisfied'] = ""
        gr.Warning("Since you are not quite satisfied. " \
                   "Please leave some comments!")
        return [user_data, 
                gr.update(),  # satisfaction
                gr.update(),  # why_unsatisfied
                gr.update(),  # save_button
                gr.update()  # generate_button
        ]
    
    #********** save to csv ***************************************************
    store_fpath = config.user_data_store_fpath
    # Fetch the last timestamp from the CSV file if it exists
    last_timestamp = None
    if os.path.exists(store_fpath):
        with open(store_fpath, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] != user_data['username']:
                    continue
                last_timestamp = row[1]  # Assuming the second column is the timestamp

    # Determine the new timestamp
    if last_timestamp is None:
        last_timestamp = 0
    else:
        last_timestamp = int(last_timestamp) + 1

    user_data['timestamp'] = last_timestamp

    # Prepare data in the specified format
    row = [
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
    gr.Info("Successfully saved info. If you want to save the image, "\
            "please click the button on the top right of image")
    #*************************************************************
    
    user_data["satisfaction"] = 1
    user_data['why_unsatisfied'] = ""
    
    final_return = [
        user_data, 
        gr.update(visible=False, value=None), # satisfaction
        gr.update(visible=False, value=""), # why_unsatisfied
        gr.update(visible=False), # save_button
        gr.update(visible=True), # generate_button
    ]
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

def disable_ui():
    final_return = []
    for i in range(8):
        final_return.append(gr.update(interactive=False))
    return final_return

def enable_ui():
    final_return = []
    for i in range(8):
        final_return.append(gr.update(interactive=True))
    return final_return



################################################################################

def clear_cuda_memory(loaded_model, user_data):
    username = user_data['username']
    MYLOGGER.info(f"---- USER {username}: ready to clear cuda memory")
    if loaded_model is None: return
    
    device = loaded_model.device   
    MYLOGGER.info(f"---- USER {username}: model was on device: {device}")
    
    before = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
    before /= (1024 ** 2)
    
    gc.collect()
    with torch.no_grad(): 
        with torch.cuda.device(device):
            torch.cuda.empty_cache()   
    del loaded_model
    
    after = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
    after /= (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: GPU memory released {(after - before):.2f} MB")

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
    
def find_most_idle_gpu(estimated_model_size):
    num_gpus = torch.cuda.device_count()
    max_free_memory = 0
    best_gpu = None
    for i in range(num_gpus):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        free_memory = total_memory - allocated_memory
        # Check if the GPU has enough memory for the model
        if free_memory >= estimated_model_size and free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i
    if best_gpu is None:
        gr.Warning("No GPU with enough free memory for the model. Waiting for idle GPU")
    return best_gpu

def wait_for_idle_gpu(estimated_model_size, check_interval=6):
    while True:
        best_gpu = find_most_idle_gpu(estimated_model_size)
        if best_gpu is not None:
            return best_gpu
        time.sleep(check_interval)

def estimate_model_size(user_data):
    if user_data['image_size'] == 512:
        return 7.5e+9
    else:
        return 1.2e+10

def load_model(user_data):
    """
    Returns:
        loaded_model
    """
    username = user_data['username']
    
    image_style = user_data['image_style']
    image_size = user_data['image_size']

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
    
    estimated_model_size = estimate_model_size(user_data)
    
    # best_gpu = find_most_idle_gpu(estimated_model_size)
    best_gpu = wait_for_idle_gpu(estimated_model_size)

    device = torch.device(f'cuda:{best_gpu}')
    before = torch.cuda.get_device_properties(best_gpu).total_memory - torch.cuda.memory_allocated(best_gpu)
    before /= (1024 ** 2)
    #*********************
    loaded_model.to(device)
    
    after = torch.cuda.get_device_properties(best_gpu).total_memory - torch.cuda.memory_allocated(best_gpu)
    after /= (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: model loaded on device {best_gpu}")
    MYLOGGER.info(f"---- USER {username}: model usage of cuda: {(before - after):2f} MB")
    return loaded_model

def load_and_generate_image(user_data):
    """Generate images from loaded model based on user input

    Args:
        loaded_model: gr.State(StableDiffusionPipeline)
    Returns:
        image: gr.Image()
    """
    MYLOGGER.info(f"---- USER {user_data['username']} Generating image...")
    MYLOGGER.info(f"**** USER {user_data['username']}:  "\
                  f"style: {user_data['image_style']} || size: {user_data['image_size']}"\
                  f"p: {user_data['prompt']} || np: {user_data['negative_prompt']} || "\
                  f"sampling: {user_data['sampling_steps']} || cfg: {user_data['cfg_scale']} || "\
                  f"seed: {user_data['seed']}")
    loaded_model = load_model(user_data)

    image = loaded_model(
        prompt=user_data['prompt'],
        negative_prompt=user_data['negative_prompt'], 
        num_inference_steps=user_data['sampling_steps'], # sampling steps
        guidance_scale=user_data['cfg_scale'], # cfg
        generator=torch.manual_seed(user_data['seed']),
    )
    nsfw_detected = image['nsfw_content_detected'][0]
    
    image = image.images[0] # PIL.Image.Image
    
    return image, loaded_model, nsfw_detected




