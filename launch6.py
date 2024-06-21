import gc, json

import torch, uvicorn
import gradio as gr
import logging as logger
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

import config
from config import IMAGE_STYLE_CHOICES, IMAGE_SIZE_CHOICES, \
                   INITIAL_SAMPLING_STEPS, INITIAL_CFG, INITIAL_SEED, INITIAL_IMAGE_SIZE, \
                   CONCURRENCY_LIMIT
from utils.launch_utils6 import create_greeting, get_cuda_info, get_auth_cred, \
                                debug_fn, handle_save, \
                                handle_generation, satisfaction_slider_change, \
                                disable_component, update_input, \
                                MYLOGGER
                                

MYLOGGER.setLevel(logger.INFO)

app = FastAPI()

MYLOGGER.info(f"*********** CONCURRENCY_LIMIT {CONCURRENCY_LIMIT}")

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
        "satisfaction": 0,
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
            prompt = gr.Textbox(label="Prompt", lines=2, max_lines=3, 
                                placeholder="Enter your prompt here...",
                                info="Input limited to 77 words!")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, max_lines=3, 
                                            placeholder="Enter the things you don't want here...",
                                            info="Input limited to 77 words!")
            sampling_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=120, 
                                        step=1, value=INITIAL_SAMPLING_STEPS, 
                                        info="More steps usually lead to a higher quality " \
                                            "image, but may take longer time to process. Bigger value doesn't mean best!")
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, 
                                  step=1, value=INITIAL_CFG,
                                    info="A higher CFG scale value encourages the model to " \
                                        "generate images closely linked to the text, but may " \
                                        "take longer time to process. Bigger value doesn't mean best!")
            seed = gr.Number(label="Seed", value=INITIAL_SEED, precision=0,
                                info="Random initialization to the model. " \
                                    "Same seed with other same input can give the same output. " \
                                    "Please don't give extremely large number.")

            generate_button = gr.Button("Generate", interactive=True)
            # tmp_html = gr.HTML(JS_CODE)
            
            # Add a spacer to keep distance between buttons
            gr.HTML("<br><br>")
            logout_button = gr.Button("Logout", 
                                      link=f"/logout?user_id={user_data.value['username']}")
            debug = gr.Button("debug", visible=False)

        with gr.Column(scale=1):
            image_output = gr.Image(label="Output Image", interactive=False)
            satisfaction = gr.Slider(label="Satisfaction", minimum=0, maximum=7,
                                     info="Choose between 1 (unsatisfied) and 7 (satisfied)",
                                     step=1, value=0, visible=False)
            why_unsatisfied = gr.Textbox(label="What makes you unsatisfied", 
                                         lines=2, max_lines=4, visible=False)
            save_button = gr.Button("Rate It & Save Info & Generate Next", visible=False, interactive=False)
            info_output = gr.Json(label="Generation Info")
            
    ########################################################################
    
    group_ui = [image_style, image_size, prompt, negative_prompt, sampling_steps, cfg_scale, seed] # 7
    post_ui = [image_output, info_output, save_button, satisfaction, why_unsatisfied] # 5

    satisfaction.change(satisfaction_slider_change, inputs=[satisfaction], outputs=save_button)
    
    generate_button.click(fn=disable_component, inputs=group_ui + [generate_button, logout_button], 
                          outputs=group_ui + [generate_button, logout_button],
                          trigger_mode="once") \
                    .then(fn=update_input, inputs=[user_data] + group_ui,
                             outputs=[user_data],
                             trigger_mode="once") \
                    .then(fn=handle_generation, inputs=[user_data], 
                            outputs=post_ui + group_ui + [generate_button, logout_button],
                            show_progress=True, trigger_mode="once",
                            concurrency_limit=CONCURRENCY_LIMIT) \

    save_button.click(fn=handle_save, 
                      inputs=[user_data, satisfaction, why_unsatisfied],
                      outputs=[user_data, satisfaction, why_unsatisfied, save_button, generate_button] \
                                + group_ui + [info_output],
                      trigger_mode="once", show_progress=False)
    
    debug.click(fn=debug_fn, inputs=[user_data, loaded_model], trigger_mode="once")

    demo.load(create_greeting, inputs=[user_data], 
                               outputs=[user_data, login_message])

    
    def clear_cuda_mem():
        # Clear CUDA memory
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA memory cleared.")
    demo.unload(lambda : print(f"<<<<<<<< USER Logout."))
    # demo.unload(lambda : print(f"<<<<<<<< USER Logout."))

get_cuda_info()

with open(config.auth_message_fpath, 'r') as f:
    auth_message = f.read()

app, _, _ = demo.queue().launch(
    share=True,
    auth=get_auth_cred,
    auth_message=auth_message,
    max_threads=CONCURRENCY_LIMIT,
    server_name="0.0.0.0",
    server_port=7860,
)

