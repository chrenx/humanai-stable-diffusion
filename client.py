import gradio as gr
import logging as logger
from fastapi import FastAPI

from config import IMAGE_STYLE_CHOICES, IMAGE_SIZE_CHOICES, \
                   INITIAL_SAMPLING_STEPS, INITIAL_CFG, INITIAL_SEED, \
                   CONCURRENCY_LIMIT, AUTH_MSG_FPATH
from utils.client_util import load_app, get_cuda_info, register_page, login_page, forget_page, \
                                debug_fn, handle_save, \
                                satisfaction_slider_change, \
                                go_to_page_1, go_to_page_2, update_input, \
                                MYLOGGER, handle_generation
                                
MYLOGGER.setLevel(logger.INFO)

app = FastAPI()

MYLOGGER.info(f"*********** CONCURRENCY_LIMIT {CONCURRENCY_LIMIT}")


def create_generation_function(name, params):
    # Define the function dynamically with parameters
    function_code = f"""
def {name}({params}):
    return handle_generation({params})
"""
    # Execute the function definition
    exec(function_code, globals())



with gr.Blocks() as demo: 
    user_data = gr.State({
        "username": None,
        "image_style": None,
        "image_size": IMAGE_SIZE_CHOICES[0],
        "prompt": "",
        "negative_prompt": "",
        "sampling_steps": INITIAL_SAMPLING_STEPS,
        "cfg_scale": INITIAL_CFG,
        "seed": INITIAL_SEED,
        "satisfaction_0": 0,
        "satisfaction_1": 0,
        "satisfaction_2": 0,
        "satisfaction_3": 0,
        "comment_0": "",
        "comment_1": "",
        "comment_2": "",
        "comment_3": "",
        "proper_save": [False]*4,
    })
    
    ########################################################################
    image_style_btns = []
    
    with gr.Row():
        login_message = gr.Markdown(value="Not logged in")

    with gr.Row(visible=False) as page_1:
        with gr.Column(scale=1) as input_col:
            num_images = gr.Dropdown(label="Number of images you want to generate for one prompt",
                                    choices=[1,2,3,4], value=2,
                                    info="The larger value, the longer it's gonna take to generate")
            image_size = gr.Dropdown(label="Image Resolution (Width x Height)",
                                        choices=IMAGE_SIZE_CHOICES, value=IMAGE_SIZE_CHOICES[0])
            prompt = gr.Textbox(label="Prompt", lines=2, max_lines=3, 
                                placeholder="Enter your prompt here...",
                                info="Input limited to 77 words!")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, max_lines=3, 
                                            placeholder="Enter the things you don't want here...",
                                            info="Input limited to 77 words!")
            sampling_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=100, 
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
   
            # Add a spacer to keep distance between buttons
            gr.HTML("<br><br>")
            logout_button = gr.Button("Logout", 
                                      link=f"/logout?user_id={user_data.value['username']}")
            debug = gr.Button("debug", visible=False)

        with gr.Column(scale=1):
            with gr.Row():
                for i in range(len(IMAGE_STYLE_CHOICES)):
                    image_style = IMAGE_STYLE_CHOICES[i]
                    button = gr.Button(image_style, interactive=True)
                    image_style_btns.append(button)
                    create_generation_function(f"generation_{i}", "params")
            
    with gr.Row(visible=False) as page_2:
        with gr.Column(visible=True, scale=1, variant='panel') as col_image_0:
            image_output_0 = gr.Image(label="Output Image 1", interactive=False)
            satisfaction_0 = gr.Slider(label="Satisfaction", minimum=0, maximum=7,
                                    info="Choose between 1 (unsatisfied) and 7 (satisfied)",
                                    step=1, value=0, visible=False)
            comment_0 = gr.Textbox(label="What makes you unsatisfied", 
                                        lines=2, max_lines=4, visible=False)
            
        with gr.Column(visible=True, scale=1, variant='panel') as col_image_1:
            image_output_1 = gr.Image(label="Output Image 2", interactive=False)
            satisfaction_1 = gr.Slider(label="Satisfaction", minimum=0, maximum=7,
                                    info="Choose between 1 (unsatisfied) and 7 (satisfied)",
                                    step=1, value=0, visible=False)
            comment_1 = gr.Textbox(label="What makes you unsatisfied", 
                                        lines=2, max_lines=4, visible=False)
            
        with gr.Column(visible=True, scale=1, variant='panel') as col_image_2:
            image_output_2 = gr.Image(label="Output Image 3", interactive=False)
            satisfaction_2 = gr.Slider(label="Satisfaction", minimum=0, maximum=7,
                                    info="Choose between 1 (unsatisfied) and 7 (satisfied)",
                                    step=1, value=0, visible=False)
            comment_2 = gr.Textbox(label="What makes you unsatisfied", 
                                        lines=2, max_lines=4, visible=False)
        
        with gr.Column(visible=True, scale=1, variant='panel') as col_image_3:
            image_output_3 = gr.Image(label="Output Image 4", interactive=False)
            satisfaction_3 = gr.Slider(label="Satisfaction", minimum=0, maximum=7,
                                    info="Choose between 1 (unsatisfied) and 7 (satisfied)",
                                    step=1, value=0, visible=False)
            comment_3 = gr.Textbox(label="What makes you unsatisfied", 
                                        lines=2, max_lines=4, visible=False)
        
    with gr.Row(visible=False) as page_2_2:
        save_button = gr.Button("Rate It & Save Info", 
                                visible=True, interactive=False)
        generate_next = gr.Button("Generate Next", 
                                visible=True, interactive=False)
        
    with gr.Row(visible=False) as page_2_3:
        info_output = gr.Json(label="Generation Info")
            
    # Authentification #########################################################            
    with gr.Row(visible=True) as page_auth:
        with gr.Tab("Login"):
            login_username = gr.Textbox(label="Email")
            login_password = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Login")

            login_button.click(fn=login_page, inputs=[login_username, login_password, user_data], 
                               outputs=[login_message, page_auth, page_1, user_data],
                               show_progress=False)
        
        with gr.Tab("Register"):
            register_email = gr.Textbox(label="Email")
            register_button = gr.Button("Register")
        
            register_button.click(fn=register_page, inputs=[register_email],
                                  show_progress=False)
            
        with gr.Tab("Forget Account"):
            forget_email = gr.Textbox(label="Email")
            forget_button = gr.Button("Get New Password")
            
            forget_button.click(fn=forget_page, inputs=[forget_email],
                                show_progress=False)
    ########################################################################
    
    group_ui = [num_images, image_size, prompt, negative_prompt, sampling_steps, cfg_scale, seed] #7

    all_col_image = [col_image_0, col_image_1, col_image_2, col_image_3]
    all_image_output = [image_output_0, image_output_1, image_output_2, image_output_3]
    
    all_satisfaction = [satisfaction_0, satisfaction_1, satisfaction_2, satisfaction_3]
    all_comment = [comment_0, comment_1, comment_2, comment_3]


    satisfaction_0.change(satisfaction_slider_change, 
                          inputs=[user_data] + all_satisfaction, outputs=save_button)
    satisfaction_1.change(satisfaction_slider_change, 
                          inputs=[user_data] + all_satisfaction, outputs=save_button)
    satisfaction_2.change(satisfaction_slider_change, 
                          inputs=[user_data] + all_satisfaction, outputs=save_button)
    satisfaction_3.change(satisfaction_slider_change, 
                          inputs=[user_data] + all_satisfaction, outputs=save_button)

    
    for i in range(len(image_style_btns)):
        image_style = IMAGE_STYLE_CHOICES[i]
        btn = image_style_btns[i]
        btn.click(fn=update_input, inputs=[user_data, gr.State(image_style)] + group_ui,
                                   outputs=[user_data],
                                   trigger_mode="once") \
            .then(fn=go_to_page_2,
                                   outputs=[page_1, page_2, page_2_2, page_2_3] + all_col_image,
                                   trigger_mode="once") \
            .then(fn=globals()[f"generation_{i}"], 
                        inputs=[user_data], 
                        outputs=[user_data, page_1, page_2, page_2_2, page_2_3, info_output] + \
                                all_col_image + all_image_output + all_satisfaction + all_comment,
                        show_progress=True, trigger_mode="once",
                        concurrency_limit=1, queue=True) \

    save_button.click(fn=handle_save, 
                      inputs=[user_data] + all_satisfaction + all_comment,
                      outputs=[user_data, info_output, save_button, generate_next],
                      trigger_mode="once", show_progress=False)
    
    generate_next.click(fn=go_to_page_1, 
                      outputs=[page_1, page_2, page_2_2, page_2_3, save_button, generate_next] + \
                              all_image_output + all_satisfaction + all_comment,
                      trigger_mode="once", show_progress=False)
    
    debug.click(fn=debug_fn, inputs=[user_data], trigger_mode="once")

    demo.load(load_app, inputs=[user_data], 
                               outputs=[user_data, login_message, page_auth,
                                        page_1, page_2, page_2_2, page_2_3])

####################################################################################################

get_cuda_info()

with open(AUTH_MSG_FPATH, 'r') as f:
    auth_message = f.read()

app, _, _ = demo.queue().launch(
    share=True,
    # auth=get_auth_cred,
    # auth_message=auth_message,
    max_threads=CONCURRENCY_LIMIT,
    server_name="0.0.0.0",
    server_port=7860,
)

