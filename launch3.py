import gc, json

import torch, uvicorn
import gradio as gr
import logging as logger
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from flask import session

import config
from config import CRED_FPATH, IMAGE_STYLE_CHOICES, IMAGE_SIZE_CHOICES, auth_message_fpath
from utils.launch_utils3 import create_greeting, get_cuda_info, \
                                update_cfg_scale, update_image_size, update_image_style, \
                                update_negative_prompt, update_prompt, \
                                update_sampling_steps, update_seed, \
                                handle_generate, handle_save, MYLOGGER

MYLOGGER.setLevel(logger.INFO)

app = FastAPI()

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

def handle_logout(user_data, loaded_model, gr_request: gr.Request):
    """Clear everything after logout
    Returns:
        logined_username: gr.State()
        user_data: gr.State({})
        loaded_model: gr.State()
        image_style: gr.Dropdown()
    """
    print('........handling logout')
    if loaded_model is not None:
        clear_cuda_memory(loaded_model)
        
    cur_url = gr_request.headers['origin']
    
    response = RedirectResponse(url="/logout")
    

    print(f"User {user_data['username']} Logout")

    return None, {}, None, None
    # return response


get_cuda_info()

    
with open(config.auth_message_fpath, 'r') as f:
    auth_message = f.read()

with gr.Blocks() as demo: 
    user_data = gr.State({
        "username": None,
        "image_style": None,
        "image_size": 512,
        "prompt": "",
        "negative_prompt": "",
        "sampling_steps": 20,
        "cfg_scale": 5,
        "seed": 246372,
        "satisfaction": 1,
        "why_unsatisfied": "",
    })
    
    MYLOGGER.info("----------------------------")
    print(user_data.value)
    
    ########################################################################
    with gr.Row():
        login_message = gr.Markdown(value="Not logged in")
    with gr.Row():
        with gr.Column(scale=1):
            image_style = gr.Dropdown(label="Select Your Imaging Style", 
                                        choices=IMAGE_STYLE_CHOICES, value=None, visible=False)
            image_size = gr.Dropdown(label="Image Resolution",
                                        choices=IMAGE_SIZE_CHOICES, value=512, visible=False)
            prompt = gr.Textbox(label="Prompt", lines=2, max_lines=4, visible=False
                                placeholder="Enter your prompt here...")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, 
                                         max_lines=4, visible=False
                                         placeholder="Enter the things you don't want here...")
            sampling_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=150, 
                                        step=1, value=20, visible=False
                                        info="More steps usually lead to a higher quality " \
                                            "image, but may take longer time to process")
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=1, value=5,
                                  visible=False,
                                    info="A higher CFG scale value encourages the model to " \
                                        "generate images closely linked to the text, but may " \
                                        "take longer time to process")
            seed = gr.Number(label="Seed", value=246372, precision=0, visible=False
                                info="Random initialization to the model. " \
                                    "Same seed with other same input can give the same output. " \
                                    "Please don't give extremely large number.")

            generate_button = gr.Button("Generate", visible=False)
            # tmp_html = gr.HTML(JS_CODE)
            
            # Add a spacer to keep distance between buttons
            gr.HTML("<br><br>")
            logout_button = gr.Button("Logout", link='/logout', visible=False)
            debug = gr.Button("debug", visible=False)

        with gr.Column(scale=1):
            image_output = gr.Image(label="Output Image", visible=False)
            satisfaction = gr.Slider(label="Satisfaction", minimum=1, maximum=7,
                                     info="Choose between 1 (unsatisfied) and 7 (satisfied)",
                                     step=1, value=1, visible=False)
            why_unsatisfied = gr.Textbox(label="What makes you unsatisfied", 
                                         lines=2, max_lines=4, visible=False)
            save_button = gr.Button("Save Info & Generate Next", visible=False)
            info_output = gr.Json(label="Generation Info", visible=False)
            
    ########################################################################
            
    image_style.change(update_image_style, inputs=[user_data, image_style], 
                                           outputs=[user_data, image_style],
                                           show_progress=False)
    image_size.change(update_image_size, inputs=[user_data, image_size], 
                                         outputs=[user_data, image_size],
                                         show_progress=False)
    # prompt.change(update_prompt, inputs=[user_data, prompt],
    #                              outputs=[user_data, prompt],
    #                              show_progress=False)
    # negative_prompt.change(update_negative_prompt, inputs=[user_data, prompt],
    #                                                outputs=[user_data, prompt],
    #                                                show_progress=False)
    # sampling_steps.change(update_sampling_steps, inputs=[user_data, sampling_steps],
    #                                              outputs=[user_data, sampling_steps],
    #                                              show_progress=False)
    # cfg_scale.change(update_cfg_scale, inputs=[user_data, cfg_scale],
    #                                    outputs=[user_data, cfg_scale],
    #                                    show_progress=False)
    # seed.change(update_seed, inputs=[user_data, seed],
    #                          outputs=[user_data, seed],
    #                          show_progress=False)
    
    generate_button.click(fn=handle_generate, 
                          inputs=[user_data, prompt, negative_prompt, sampling_steps, cfg_scale, seed], 
                          outputs=[user_data, image_output, satisfaction, 
                                   why_unsatisfied, save_button, info_output,
                                   generate_button],
                          show_progress=True, queue=True,
                          trigger_mode="once")
    
    save_button.click(fn=handle_save,
                      inputs=[user_data, satisfaction, why_unsatisfied],
                      outputs=[user_data, satisfaction, why_unsatisfied, 
                               save_button, generate_button])
    
    # logout_button.click(fn=handle_logout, 
    #                     inputs=[user_data, loaded_model],
    #                     outputs=[logined_username, user_data, loaded_model, image_style],)
                        # js="clearCookiesAndLogout")
                        # outputs=[])
                        
    debug.click(fn=debug_fn, inputs=[user_data])

    demo.load(create_greeting, inputs=[user_data], 
                    outputs=[user_data, login_message, image_style, image_size, 
                             prompt, negative_prompt, sampling_steps, cfg_scale, 
                             seed, generate_button, logout_button, 
                             image_output, info_output])

    demo.unload(lambda: print("unload"))

demo.queue(max_size=23)

@app.get("/logout")
def logout(request: Request):
    # clear_cuda_memory(loaded_model.value)
    response = RedirectResponse(url="/")
    cookies = request.cookies
    for cookie in cookies:
        if cookie.startswith('access-token'):
            response.delete_cookie(cookie)
    gc.collect()
    torch.cuda.empty_cache()
    MYLOGGER.info("Log out.")
    return response

app = gr.mount_gradio_app(app, demo, path='/',
                            auth=get_auth_cred, auth_message=auth_message)
uvicorn.run(app)