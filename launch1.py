import gc, json

import torch, uvicorn
import gradio as gr
import logging as logger
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

import config
from config import IMAGE_STYLE_CHOICES, IMAGE_SIZE_CHOICES
from utils.launch_utils1 import create_greeting, get_cuda_info, get_auth_cred, \
                                update_image_size, update_image_style, debug_fn, \
                                handle_generate, handle_save, disable_ui, enable_ui, \
                                MYLOGGER

MYLOGGER.setLevel(logger.INFO)

app = FastAPI()

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
    
    ########################################################################
    with gr.Row():
        login_message = gr.Markdown(value="Not logged in")
    with gr.Row():
        with gr.Column(scale=1):
            image_style = gr.Dropdown(label="Select Your Imaging Style", 
                                        choices=IMAGE_STYLE_CHOICES, value=None)
            image_size = gr.Dropdown(label="Image Resolution",
                                        choices=IMAGE_SIZE_CHOICES, value=512)
            prompt = gr.Textbox(label="Prompt", lines=2, max_lines=4, 
                                placeholder="Enter your prompt here...")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, max_lines=4, 
                                            placeholder="Enter the things you don't want here...")
            sampling_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=150, 
                                        step=1, value=25, 
                                        info="More steps usually lead to a higher quality " \
                                            "image, but may take longer time to process")
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=1, value=7,
                                    info="A higher CFG scale value encourages the model to " \
                                        "generate images closely linked to the text, but may " \
                                        "take longer time to process")
            seed = gr.Number(label="Seed", value=246372, precision=0,
                                info="Random initialization to the model. " \
                                    "Same seed with other same input can give the same output. " \
                                    "Please don't give extremely large number.")

            generate_button = gr.Button("Generate")
            # tmp_html = gr.HTML(JS_CODE)
            
            # Add a spacer to keep distance between buttons
            gr.HTML("<br><br>")
            logout_button = gr.Button("Logout", 
                                      link=f"/logout?user_id={user_data.value['username']}")
            debug = gr.Button("debug", visible=False)

        with gr.Column(scale=1):
            image_output = gr.Image(label="Output Image")
            satisfaction = gr.Slider(label="Satisfaction", minimum=1, maximum=7,
                                     info="Choose between 1 (unsatisfied) and 7 (satisfied)",
                                     step=1, value=1, visible=False)
            why_unsatisfied = gr.Textbox(label="What makes you unsatisfied", 
                                         lines=2, max_lines=4, visible=False)
            save_button = gr.Button("Save Info & Generate Next", visible=False)
            info_output = gr.Json(label="Generation Info")
            
    ########################################################################
            
    image_style.change(update_image_style, inputs=[user_data, image_style], 
                                           outputs=[user_data, image_style],
                                           show_progress=False)
    image_size.change(update_image_size, inputs=[user_data, image_size], 
                                         outputs=[user_data, image_size],
                                         show_progress=False)
    
    generate_button.click(fn=disable_ui, outputs=[image_style, image_size, prompt, 
                                                  negative_prompt, sampling_steps, 
                                                  cfg_scale, seed, generate_button])\
                   .then(fn=handle_generate,
                         inputs=[user_data, prompt, negative_prompt, 
                                 sampling_steps, cfg_scale, seed], 
                         outputs=[user_data, image_output, satisfaction, why_unsatisfied, 
                                  save_button, info_output, generate_button],
                         show_progress=True, trigger_mode="once")\
                   .then(fn=enable_ui, outputs=[image_style, image_size, prompt, 
                                                negative_prompt, sampling_steps, 
                                                cfg_scale, seed, generate_button])
    
    save_button.click(fn=handle_save,
                      inputs=[user_data, satisfaction, why_unsatisfied],
                      outputs=[user_data, satisfaction, why_unsatisfied, 
                               save_button, generate_button])
    
    debug.click(fn=debug_fn, inputs=[user_data])

    demo.load(create_greeting, inputs=[user_data], 
                               outputs=[user_data, login_message])
    demo.unload(lambda : print(f"<<<<<<<< USER Logout."))


get_cuda_info()

with open(config.auth_message_fpath, 'r') as f:
    auth_message = f.read()

app, _, _ = demo.queue(max_size=10, default_concurrency_limit=10).launch(
    share=True,
    auth=get_auth_cred,
    auth_message=auth_message,
)

# @app.get("/logout")
# def logout(request: Request, user_id: str = None):
#     # clear_cuda_memory(loaded_model.value)
#     print("他来了ss")
#     response = RedirectResponse(url="/")
#     cookies = request.cookies
#     for cookie in cookies:
#         if cookie.startswith('access-token'):
#             response.delete_cookie(cookie)
#     gc.collect()
#     torch.cuda.empty_cache()
#     MYLOGGER.info(f"<<<<<<<< USER {user_id} Logout.")
#     return response
