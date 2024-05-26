import gc, json

import torch, uvicorn
import gradio as gr
import logging as logger
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

import config
from config import IMAGE_STYLE_CHOICES, IMAGE_SIZE_CHOICES, \
                   INITIAL_SAMPLING_STEPS, INITIAL_CFG, INITIAL_SEED, INITIAL_IMAGE_SIZE, \
                   NUM_GPU, NUM_USER_PER_GPU
from utils.launch_utils1 import create_greeting, get_cuda_info, get_auth_cred, \
                                update_image_size, update_image_style, debug_fn, \
                                handle_generate, handle_save, pre_gen_ui, post_gen_ui, \
                                handle_load_model, handle_model2device, \
                                handle_image_generation, handle_init_model, \
                                MYLOGGER

MYLOGGER.setLevel(logger.INFO)

app = FastAPI()

with gr.Blocks() as demo: 
    user_data = gr.State({
        "username": None,
        "image_style": None,
        "image_size": INITIAL_IMAGE_SIZE,
        "prompt": "",
        "negative_prompt": "",
        "sampling_steps": INITIAL_SAMPLING_STEPS,
        "cfg_scale": INITIAL_CFG,
        "seed": INITIAL_SEED,
        "satisfaction": 1,
        "why_unsatisfied": "",
    })
    loaded_model = gr.State(None)
    gen_stop_flag = gr.State(False)
    
    MYLOGGER.info("----------------------------")
    
    ########################################################################
    with gr.Row():
        login_message = gr.Markdown(value="Not logged in")
    with gr.Row():
        with gr.Column(scale=1):
            image_style = gr.Dropdown(label="Select Your Imaging Style", 
                                        choices=IMAGE_STYLE_CHOICES, value=None)
            image_size = gr.Dropdown(label="Image Resolution (currently only support 512)",
                                        choices=IMAGE_SIZE_CHOICES, value=512)
            prompt = gr.Textbox(label="Prompt", lines=2, max_lines=4, 
                                placeholder="Enter your prompt here...")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, max_lines=4, 
                                            placeholder="Enter the things you don't want here...")
            sampling_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=150, 
                                        step=1, value=INITIAL_SAMPLING_STEPS, 
                                        info="More steps usually lead to a higher quality " \
                                            "image, but may take longer time to process")
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, 
                                  step=1, value=INITIAL_CFG,
                                    info="A higher CFG scale value encourages the model to " \
                                        "generate images closely linked to the text, but may " \
                                        "take longer time to process")
            seed = gr.Number(label="Seed", value=INITIAL_SEED, precision=0,
                                info="Random initialization to the model. " \
                                    "Same seed with other same input can give the same output. " \
                                    "Please don't give extremely large number.")

            generate_button = gr.Button("Generate")
            # tmp_html = gr.HTML(JS_CODE)
            
            # Add a spacer to keep distance between buttons
            gr.HTML("<br><br>")
            logout_button = gr.Button("Logout", 
                                      link=f"/logout?user_id={user_data.value['username']}")
            debug = gr.Button("debug", visible=True)

        with gr.Column(scale=1):
            image_output = gr.Image(label="Output Image", interactive=False)
            satisfaction = gr.Slider(label="Satisfaction", minimum=1, maximum=7,
                                     info="Choose between 1 (unsatisfied) and 7 (satisfied)",
                                     step=1, value=1, visible=False)
            why_unsatisfied = gr.Textbox(label="What makes you unsatisfied", 
                                         lines=2, max_lines=4, visible=False)
            save_button = gr.Button("Save Info & Generate Next", visible=False)
            info_output = gr.Json(label="Generation Info")
            
    ########################################################################
            
    image_style.change(handle_init_model, inputs=[user_data, image_style], 
                                           outputs=[user_data, image_style],
                                           show_progress=False)
    # image_size.change(update_image_size, inputs=[user_data, image_size], 
    #                                      outputs=[user_data, image_size],
    #                                      show_progress=False)
    
    generate_button.click(fn=pre_gen_ui, 
                          inputs=[user_data, image_style, image_size, prompt, 
                                  negative_prompt, sampling_steps, 
                                  cfg_scale, seed],
                          outputs=[user_data, image_style, image_size, prompt, 
                                   negative_prompt, sampling_steps, 
                                   cfg_scale, seed, generate_button, gen_stop_flag],
                          show_progress=True,)\
                   .success(fn=handle_load_model,
                         inputs=[user_data, gen_stop_flag], 
                         outputs=[loaded_model, gen_stop_flag],
                         show_progress=True, trigger_mode="once")\
                   .success(fn=handle_model2device,
                         inputs=[user_data, loaded_model, gen_stop_flag],
                         outputs=[loaded_model, gen_stop_flag],
                         show_progress=True, trigger_mode="once")\
                   .success(fn=handle_image_generation, # and offload model as well
                         inputs=[user_data, loaded_model, gen_stop_flag],
                         outputs=[image_output, info_output, loaded_model, 
                                  image_style, image_size, prompt, 
                                  negative_prompt, sampling_steps, 
                                  cfg_scale, seed, generate_button, 
                                  gen_stop_flag],
                         show_progress=True, trigger_mode="once")\
                   .success(fn=post_gen_ui, 
                         inputs=[user_data, gen_stop_flag],
                         outputs=[save_button, satisfaction, why_unsatisfied, gen_stop_flag])
    
    save_button.click(fn=handle_save,
                      inputs=[user_data, satisfaction, why_unsatisfied],
                      outputs=[user_data, satisfaction, why_unsatisfied, 
                               save_button, generate_button])
    
    debug.click(fn=debug_fn, inputs=[user_data, loaded_model])

    demo.load(create_greeting, inputs=[user_data], 
                               outputs=[user_data, login_message])
    # demo.unload(lambda : print(f"<<<<<<<< USER Logout."))

get_cuda_info()

with open(config.auth_message_fpath, 'r') as f:
    auth_message = f.read()

concurrency_limit = NUM_GPU * NUM_USER_PER_GPU
app, _, _ = demo.queue(default_concurrency_limit=concurrency_limit).launch(
    share=True,
    auth=get_auth_cred,
    auth_message=auth_message,
)
