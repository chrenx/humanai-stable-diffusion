import argparse, json, requests, uvicorn, gc, torch

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from flask import session

from utils.launch_utils2 import change_image_size, clear_cuda_memory, \
                               generate_image, get_cuda_info, load_model


IMAGE_STYLE_CHOICES = ["Oil Painting", "Watercolor", "MoXin (traditional Chinese Painting)"]
IMAGE_SIZE_CHOICES = [512, 1024]
JS_CODE = """
function addScript() {
    var script = document.createElement("script");
    script.type = "text/javascript";
    script.text = `
        function goLogout() {
            document.getElementById('gologout').click();
        }
    `;
    document.head.appendChild(script);
}
"""

app = FastAPI()


def debug_fn(logined_username, user_data, loaded_model):
    print('logined_username: ', logined_username)
    print('user_data: ', user_data)
    print('loaded_model: ', loaded_model)

def handle_generate(user_data, image_style, loaded_model, prompt, 
                    negative_prompt, sampling_steps, cfg_scale, seed):
    """
    Returns:
        user_data: gr.State()
        image_output: gr.Image()
        info_output: gr.Json()
    """
    
    if image_style is None:
        gr.Warning("Please select your imaging style!")
        return user_data, None, user_data
    
    if prompt == '' and negative_prompt == '' or prompt.isspace() and negative_prompt.isspace():
        gr.Warning("Please provide some prompt!")
        return user_data, None, user_data
    
    if prompt.isspace(): prompt = ""
    if negative_prompt.isspace(): negative_prompt = ""
    
    user_data['image_style'] = image_style
    user_data['prompt'] = prompt
    user_data['negative_prompt'] = negative_prompt
    user_data['sampling_steps'] = sampling_steps
    user_data['cfg_scale'] = cfg_scale
    user_data['seed'] = int(seed)
    user_data['image_size'] = 512
    
    print("======================")
    print(user_data)
    print("======================")

    image, info = generate_image(loaded_model, user_data)
    
    if image is None:
        gr.Warning("Something wrong with the image generation...")
        return user_data, None, user_data
    return user_data, image, info

def get_auth_cred(username, password):
    """
    Return True if username and password match in the credential file.
    """
    with open(CRED_FPATH, encoding='utf-8') as f:
        cred = json.load(f)
    if username not in cred or cred[username] != password:
        return False
    return True

def create_greeting(logined_username, user_data, request: gr.Request):
    logined_username = request.username
    user_data['username'] = logined_username
    return logined_username, user_data, f"Welcome to Text2Image Generation,  {logined_username}!"

def trigger_javascript():
    return "goLogout()"


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


    
parser = argparse.ArgumentParser()

parser.add_argument("--auth_cred_fpath", type=str, default="auth.json",
                                        help="user credentials")

parser.add_argument("--auth_message_fpath", type=str, default="data_agreement.html",
                                            help="data agreement file path")

parser.add_argument("--not_share_link", action="store_true", 
                                        help="Do not generate sharable link when turned on")
args = parser.parse_args()

global CRED_FPATH
CRED_FPATH = args.auth_cred_fpath

get_cuda_info()

    
with open(args.auth_message_fpath, 'r') as f:
    auth_message = f.read()

with gr.Blocks() as demo: 
    logined_username = gr.State(None)
    user_data = gr.State({"image_size": 512})
    loaded_model = gr.State(None)
    
    print("----------------------------")
    print(logined_username.value)
    print(user_data.value)
    print(loaded_model.value)
    
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
                                        step=1, value=20, 
                                        info="More steps usually lead to a higher quality " \
                                            "image, but may take longer time to process")
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=1, value=7,
                                    info="A higher CFG scale value encourages the model to " \
                                        "generate images closely linked to the text, but mat " \
                                        "take longer time to process")
            seed = gr.Number(label="Seed", value=246372, precision=0,
                                info="Random initialization to the model. " \
                                    "Same seed with other same input can give the same output. " \
                                    "Please don't give extremely large number.")

            generate_button = gr.Button("Generate")
            # tmp_html = gr.HTML(JS_CODE)
            
            # Add a spacer to keep distance between buttons
            gr.HTML("<br><br>")
            logout_button = gr.Button("Logout", link='/logout')
            debug = gr.Button("debug")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Output Image")
            info_output = gr.Json(label="Generation Info")
            
    # invisible_logout_btn = gr.Button(elem_id="gologout", link="/logout", visible=False)
            
    ########################################################################
            
    image_style.change(load_model, inputs=[user_data, image_style, loaded_model, image_size], 
                                    outputs=[user_data, image_style, loaded_model],
                                    show_progress=True, queue=True, trigger_mode="once")
    image_size.change(change_image_size, inputs=[user_data, image_size, loaded_model], 
                                            outputs=[user_data, image_size, loaded_model],
                                            show_progress=True, queue=True, trigger_mode="once")

    generate_button.click(fn=handle_generate, 
                            inputs=[user_data, image_style, loaded_model, prompt, 
                                    negative_prompt, sampling_steps, cfg_scale, seed], 
                            outputs=[user_data, image_output, info_output])
    
    # logout_button.click(fn=handle_logout, 
    #                     inputs=[user_data, loaded_model],
    #                     outputs=[logined_username, user_data, loaded_model, image_style],)
                        # js="clearCookiesAndLogout")
                        # outputs=[])
                        
    debug.click(fn=debug_fn, inputs=[logined_username, user_data, loaded_model])

    demo.load(create_greeting, inputs=[logined_username, user_data], 
                                outputs=[logined_username, user_data, login_message])
    demo.unload(lambda: print("unload"))


demo.queue()

@app.get("/logout")
def logout(request: Request):
    print('asdddddddddddddddddd')
    # clear_cuda_memory(loaded_model.value)
    response = RedirectResponse(url="/")
    cookies = request.cookies
    for cookie in cookies:
        if cookie.startswith('access-token'):
            response.delete_cookie(cookie)
    print("Logout user!")
    gc.collect()
    torch.cuda.empty_cache()
    return response

app = gr.mount_gradio_app(app, demo, path='/',
                            auth=get_auth_cred, auth_message=auth_message)
uvicorn.run(app)