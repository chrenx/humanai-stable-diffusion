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
from config import MODEL_NAME_PATH_MAP, CRED_FPATH, SAFETY_CHECK, \
                   FREE_MEMORY_THRESHOLD, NUM_USER_PER_GPU

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

logger.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
MYLOGGER = logger.getLogger()

class GPUMonitor:
    def __init__(self):
        print('GPU monitor created ...')
        self.allocation = {}
        
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

        if cuda_visible_devices:
            cuda_ids = cuda_visible_devices.split(',')
            for i, cuda_id in enumerate(cuda_ids):
                self.allocation[int(i)] = {
                    "users": [],
                    "cuda_id": int(cuda_id)
                }
        else:
            for i in range(torch.cuda.device_count()): 
                self.allocation[int(i)] = {
                    "users": [],
                    "cuda_id": int(i)
                }
    
    def get_free_device(self, username):
        # if username == "2":
        #     cuda_id, device_id = 7, 1
        #     self.allocation[1]["users"].append(username)
        #     return cuda_id, device_id
        
        for device_id, info in self.allocation.items():
            if len(info["users"]) < NUM_USER_PER_GPU: #! only allow # users on one GPU
                free_memory = self._get_freememory(device_id)
                if free_memory < FREE_MEMORY_THRESHOLD:
                    continue
                self.allocation[device_id]["users"].append(username)
                cuda_id = info['cuda_id']
                MYLOGGER.info(f"---- GPU Monitor gives USER {username} cuda {cuda_id}, "\
                              f"device {device_id}")
                return cuda_id, device_id
        return None, None
    
    def release_device(self, device_id, username):
        MYLOGGER.info(f"---- USER {username} released from device {device_id}")
        self.allocation[device_id]["users"].remove(username)
        
    def _get_freememory(self, device_id):
        return torch.cuda.get_device_properties(device_id).total_memory - \
               torch.cuda.memory_allocated(device_id)
        

GPU_monitor = GPUMonitor() 

################################################################################

def debug_fn(user_data, loaded_model):
    print("\n================================")
    print('user_data: ', user_data)
    print("loaded_model: ")
    print(loaded_model)
    print()
    print("GPU MONITOR:")
    print(GPU_monitor.allocation)
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

def handle_generate(user_data, prompt, negative_prompt, 
                    sampling_steps, cfg_scale, seed, loaded_model):
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

    image, loaded_model, nsfw_detected = load_and_generate_image(user_data, loaded_model)
    
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
    
    clear_cuda_memory(loaded_model)
    
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


def handle_load_model(user_data, gen_stop_flag):
    
    if gen_stop_flag:
        return gr.update(), gr.update()
    
    MYLOGGER.info(f"---- USER {user_data['username']}: handle load model")
    
    gr.Info("Server is working load_model ...")

    safetensor_path = MODEL_NAME_PATH_MAP[user_data['image_style']]
    
    # progress(0.2, "Loading safety checker...")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    
    # progress(0.4, "Loading feature extractor...")
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # progress(0.6, "Loading model...")
    loaded_model = StableDiffusionPipeline.from_single_file(safetensor_path,
                            extract_ema=True,
                            safety_checker=safety_checker,
                            feature_extractor=feature_extractor,
                            image_size=user_data['image_size'])
    
    return loaded_model, False

def handle_model2device(user_data, loaded_model, gen_stop_flag):
    if gen_stop_flag:
        return gr.update(), gr.update()
    
    MYLOGGER.info(f"---- USER {user_data['username']}: handle model2device")
    
    gr.Info("Server is working model2device ...")
    
    username = user_data['username']
    
    cuda_id, device_id = wait_for_idle_gpu(username)
    
    # device = torch.device(f'cuda:{cuda_id}')
    before = torch.cuda.get_device_properties(device_id).total_memory - \
             torch.cuda.memory_allocated(device_id)
    before /= (1024 ** 2)
    #*********************
    loaded_model.to(f'cuda:{device_id}')
    
    after = torch.cuda.get_device_properties(device_id).total_memory - \
            torch.cuda.memory_allocated(device_id)
    after /= (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: model being put on cuda {cuda_id}, device {device_id}")
    MYLOGGER.info(f"---- USER {username}: model usage of GPU: {(before - after):2f} MB")
    return loaded_model, False
    
def handle_image_generation(user_data, loaded_model, gen_stop_flag):
    early_return = [None, None, None]
    for i in range(7):
        early_return.append(gr.update(interactive=True))
    early_return.append(gr.update(interactive=True, visible=True)) # generation button
    early_return.append(True) # gen_stop_flag
    
    if gen_stop_flag:
        return early_return
    
    username = user_data['username']
    
    MYLOGGER.info(f"---- USER {username}: handle image generation")
    
    image = loaded_model(
        prompt=user_data['prompt'],
        negative_prompt=user_data['negative_prompt'], 
        num_inference_steps=user_data['sampling_steps'], # sampling steps
        guidance_scale=user_data['cfg_scale'], # cfg
        generator=torch.manual_seed(user_data['seed']),
    )
    
    nsfw_detected = image['nsfw_content_detected'][0]
    
    image = image.images[0] # PIL.Image.Image
    
    
    # offload_model(loaded_model, user_data)
    ############################################################################
    MYLOGGER.info(f"---- USER {username}: handle offload model")
    
    device = loaded_model.device   
    MYLOGGER.info(f"---- USER {username}: model was on device: {device}")
    
    before_clear = (torch.cuda.get_device_properties(device).total_memory - \
                   torch.cuda.memory_allocated(device)) / (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: before clear ==> {before_clear}")
    
    collected = gc.collect()
    print("collected before: ", collected)
    del loaded_model
    collected = gc.collect()
    print("collected mid: ", collected)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()   
    collected = gc.collect()
    print("collected after: ", collected)
    
    after_clear = (torch.cuda.get_device_properties(device).total_memory - \
                  torch.cuda.memory_allocated(device)) / (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: after clear ==> {after_clear}")
    
    MYLOGGER.info(f"---- USER {username}: GPU memory released "\
                  f"{(after_clear - before_clear):.2f} MB")
    
    GPU_monitor.release_device(device.index, username)
    ############################################################################
    
    if SAFETY_CHECK:
        if nsfw_detected:
            MYLOGGER.info(f"---- USER {username} improper prompt")
            gr.Warning("Please use proper prompt.")
            return early_return
    
    final_return = [image, user_data, None]
    for i in range(7):
        final_return.append(gr.update(interactive=True))
    final_return.append(gr.update(visible=False)) # generation button
    final_return.append(False) # gen_stop_flag
    
    return final_return

def post_gen_ui(user_data, gen_stop_flag):
    if gen_stop_flag:
        return gr.update(), gr.update(), gr.update(), False
    
    MYLOGGER.info(f"---- USER {user_data['username']}: handle post gen ui")
    
    final_return = []
    for i in range(3):
        final_return.append(gr.update(visible=True))
    final_return.append(False)
    return final_return



def handle_init_model(user_data, image_style, image_size):
    MYLOGGER.info(f"---- USER {user_data['username']}: ---- initialize {image_style}")
    
    safetensor_path = MODEL_NAME_PATH_MAP[user_data['image_style']]
    
    safety_checker = StableDiffusionSafetyChecker\
                        .from_pretrained("CompVis/stable-diffusion-safety-checker")
    
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    loaded_model = StableDiffusionPipeline.from_single_file(safetensor_path,
                                                            extract_ema=True,
                                                            safety_checker=safety_checker,
                                                            feature_extractor=feature_extractor,
                                                            image_size=image_size)
    return loaded_model

def pre_gen_ui(user_data, image_style, image_size, prompt, negative_prompt, 
                sampling_steps, cfg_scale, seed):

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

    if user_data['image_style'] is None:
        gr.Warning("Please select your imaging style!")
        return early_return
    
    if user_data['prompt'] == '' and user_data['negative_prompt'] == '' or \
       user_data['prompt'].isspace() and user_data['negative_prompt'].isspace():
        gr.Warning("Please provide some prompt!")
        return early_return
    
    final_return = [user_data]
    for i in range(7):
        final_return.append(gr.update(interactive=False))
    final_return.append(gr.update(visible=False)) # generation button
    final_return.append(False) # gen_stop_flag
    return final_return

def handle_generation(user_data, loaded_model, image_style, image_size, prompt, 
                      negative_prompt, sampling_steps, cfg_scale, seed):
    
    user_data['image_style'] = image_style 
    user_data['image_size'] = image_size 
    user_data['prompt'] = prompt 
    user_data['negative_prompt'] = negative_prompt 
    user_data['sampling_steps'] = sampling_steps 
    user_data['cfg_scale'] = cfg_scale 
    user_data['seed'] = seed 
    
    early_return = [user_data]
    for i in range(5):
        early_return.append(gr.update())
    
    MYLOGGER.info(f"---- USER {user_data['username']}: ---- handle generation")
    
    username = user_data['username']
    
    cuda_id, device_id = GPU_monitor.get_free_device(username)
    trigger_warning = True
    
    while cuda_id is None and device_id is None:
        if trigger_warning:
            trigger_warning = False
            MYLOGGER.info(f"---- USER {username}: ---- waiting for GPU")
            gr.Warning("No GPU with enough free memory for the model. "\
                        "Waiting for idle GPU. Please don't leave.")
        cuda_id, device_id = GPU_monitor.get_free_device(username)
    
    before = (torch.cuda.get_device_properties(device_id).total_memory - \
             torch.cuda.memory_allocated(device_id)) / (1024 ** 2)

    loaded_model.to(f'cuda:{device_id}')
    
    after = (torch.cuda.get_device_properties(device_id).total_memory - \
            torch.cuda.memory_allocated(device_id)) / (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: ---- model being put on cuda" \
                                        f" {cuda_id}, device {device_id}")
    MYLOGGER.info(f"---- USER {username}: ---- model usage of GPU: {(before - after):2f} MB")
    
    image = loaded_model(
        prompt=user_data['prompt'],
        negative_prompt=user_data['negative_prompt'], 
        num_inference_steps=user_data['sampling_steps'], # sampling steps
        guidance_scale=user_data['cfg_scale'], # cfg
        generator=torch.manual_seed(user_data['seed']),
    )
    
    nsfw_detected = image['nsfw_content_detected'][0]
    
    image = image.images[0] # PIL.Image.Image
    
    collected = gc.collect()
    print("collected before: ", collected)
    del loaded_model
    collected = gc.collect()
    print("collected mid: ", collected)
    with torch.cuda.device(device_id):
        torch.cuda.empty_cache()   
    collected = gc.collect()
    print("collected after: ", collected)
    
    GPU_monitor.release_device(device_id, username)
    ############################################################################
    
    if SAFETY_CHECK:
        if nsfw_detected:
            MYLOGGER.info(f"---- USER {username} : ---- improper prompt")
            gr.Warning("Please use proper prompt!")
            return early_return
    
    final_return = [user_data, image, user_data, gr.update(visible=False)]
    for i in range(3):
        final_return.append(gr.update(visible=True))    
    
    return final_return

################################################################################

def offload_model(loaded_model, user_data):
    username = user_data['username']
    MYLOGGER.info(f"---- USER {username}: handle offload model")
    
    device = loaded_model.device   
    MYLOGGER.info(f"---- USER {username}: model was on device: {device}")
    
    before_clear = (torch.cuda.get_device_properties(device).total_memory - \
                   torch.cuda.memory_allocated(device)) / (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: before clear ==> {before_clear}")
    
    clear_cuda_memory(loaded_model)
    
    after_clear = (torch.cuda.get_device_properties(device).total_memory - \
                  torch.cuda.memory_allocated(device)) / (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: after clear ==> {after_clear}")
    
    MYLOGGER.info(f"---- USER {username}: GPU memory released "\
                  f"{(after_clear - before_clear):.2f} MB")
    
    GPU_monitor.release_device(device.index, username)

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

def wait_for_idle_gpu(username):
    trigger_warning = True
      
    while True:
        cuda_id, device_id = GPU_monitor.get_free_device(username)
        if cuda_id is not None:
            return cuda_id, device_id
        if trigger_warning:
            trigger_warning = False
            gr.Warning("No GPU with enough free memory for the model. "\
                        "Waiting for idle GPU.")

def load_model(user_data, loaded_model):
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
    
    print("[][[][]] ", loaded_model.device)
    
    cuda_id, device_id = wait_for_idle_gpu(username)
    
    # device = torch.device(f'cuda:{cuda_id}')
    before = torch.cuda.get_device_properties(device_id).total_memory - \
             torch.cuda.memory_allocated(device_id)
    before /= (1024 ** 2)
    #*********************
    loaded_model.to(f'cuda:{device_id}')
    print("now ", loaded_model.device)
    
    after = torch.cuda.get_device_properties(device_id).total_memory - \
            torch.cuda.memory_allocated(device_id)
    after /= (1024 ** 2)
    
    MYLOGGER.info(f"---- USER {username}: model loaded on cuda {cuda_id}, device {device_id}")
    MYLOGGER.info(f"---- USER {username}: model usage of GPU: {(before - after):2f} MB")
    return loaded_model

def load_and_generate_image(user_data, loaded_model):
    """Generate images from loaded model based on user input

    Args:
        loaded_model: gr.State(StableDiffusionPipeline)
    Returns:
        image: gr.Image()
    """
    gr.Info("Generation may take long time, especially when there are "\
               "multiple users using the model at the same time. "\
               "You can leave the web there and come back later!")
    MYLOGGER.info(f"---- USER {user_data['username']} Generating image...")
    MYLOGGER.info(f"**** USER {user_data['username']}:  "\
                  f"style: {user_data['image_style']} || size: {user_data['image_size']} || "\
                  f"p: {user_data['prompt']} || np: {user_data['negative_prompt']} || "\
                  f"sampling: {user_data['sampling_steps']} || cfg: {user_data['cfg_scale']} || "\
                  f"seed: {user_data['seed']}")
    loaded_model = load_model(user_data, loaded_model)

    image = loaded_model(
        prompt=user_data['prompt'],
        negative_prompt=user_data['negative_prompt'], 
        num_inference_steps=user_data['sampling_steps'], # sampling steps
        guidance_scale=user_data['cfg_scale'], # cfg
        generator=torch.manual_seed(user_data['seed']),
    )


    GPU_monitor.release_device(int(str(loaded_model.device)[5:]), user_data['username'])
    
    nsfw_detected = image['nsfw_content_detected'][0]
    
    image = image.images[0] # PIL.Image.Image
    
    return image, loaded_model, nsfw_detected




