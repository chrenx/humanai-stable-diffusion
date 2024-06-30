import socket
import threading
import pickle
from queue import Queue
from PIL import Image
import io

import torch

import warnings
from diffusers.utils import logging as diffuser_logger
import logging as LOGGER
import os

from config import IMAGE_STYLE_CHOICES
from config_preload_model import preload_all_model

diffuser_logger.set_verbosity_error()
warnings.filterwarnings("ignore")


MODEL_QUEUES = {}    # key: image style, value: Queue
MODEL_FUNCTIONS = {} # key: image style, value: generate image func

PRELOAD_MODELS = None

# Dummy image generation functions for models
def generate_image_model(user_data):
    res = {'error': False}
    try:
        LOGGER.info(f"GENERATE for USER {user_data['username']}")
        image_res = PRELOAD_MODELS[user_data['image_style']](
                        prompt=user_data['prompt'],
                        negative_prompt=user_data['negative_prompt'], 
                        num_inference_steps=user_data['sampling_steps'], # sampling steps
                        guidance_scale=user_data['cfg_scale'], # cfg
                        generator=torch.manual_seed(user_data['seed']),
                        num_images_per_prompt=user_data['num_images'],
                        width= user_data['image_width'], # 360,
                        height= user_data['image_height'], # 480
                    )
        
        # min 1, max 4
        res['num_images'] = len(image_res.images)
        for i in range(res['num_images']):
            res[f"nsfw{i}"] = image_res['nsfw_content_detected'][i]
            res[f"image{i}"] = image_res.images[i] # PIL.Image.Image

    except Exception as e:
        res['error'] = True
        LOGGER.info(f"ERROR message from USER {user_data['username']} --"\
                     "----------- generate image")
        LOGGER.info(str(e))
    finally:
        torch.cuda.empty_cache()
        return res

def register_model(image_style, generate_image_function):
    MODEL_QUEUES[image_style] = Queue()
    MODEL_FUNCTIONS[image_style] = generate_image_function

def handle_client(client_socket, addr):
    LOGGER.info(f"Connected by addr: {addr}")
    data = client_socket.recv(1024)
    if data:
        user_data = pickle.loads(data)
        LOGGER.info(f"Received request from USER {user_data['username']}:")
        for key, value in user_data.items():
            LOGGER.info(f"-----USER {user_data['username']}, {key}: {value}")
            
        if user_data['image_style'] in MODEL_QUEUES:
            # generate image
            MODEL_QUEUES[user_data['image_style']].put((client_socket, user_data))
        else:
            LOGGER.info(f"Model {user_data['image_style']} not found")
            client_socket.close()

def process_queue(image_style):
    while True:
        client_socket, user_data = MODEL_QUEUES[image_style].get()
        res = MODEL_FUNCTIONS[image_style](user_data)
        
        if not res['error']:
            # Serialize images as bytes
            for i in range(res['num_images']):
                img_byte_arr = io.BytesIO()
                res[f'image{i}'].save(img_byte_arr, format='PNG')
                res[f'image{i}'] = img_byte_arr.getvalue()

        # Pickle the dictionary and send it back to the client
        data = pickle.dumps(res)
        client_socket.sendall(data)
        LOGGER.info(f"Data sent back to the USER {user_data['username']}")

        client_socket.close()
        MODEL_QUEUES[image_style].task_done()

def main():
    LOGGER.info(f"Num of loaded models: {len(PRELOAD_MODELS.keys())}")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 65432))
    server_socket.listen(2)
    LOGGER.info("Server is waiting for connections...")

    # Register models
    for image_style in IMAGE_STYLE_CHOICES:  # Example: 10 models
        register_model(image_style, generate_image_model)
        threading.Thread(target=process_queue, args=(image_style,), daemon=True).start()

    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True).start()

if __name__ == '__main__':
    
    if os.path.exists('server.log'):
        # Remove the file
        os.remove('server.log')

    PRELOAD_MODELS = preload_all_model()
    
    # Configure logging
    # LOGGER.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
    #                 filemode='a',
    #                 format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
    #                 datefmt='%Y-%m-%d-%H:%M:%S',
    #                 filename='server.log')
    LOGGER.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
