
auth_cred_fpath = "auth.json" # user credentials
auth_message_fpath = "data_agreement.html" # data agreement file path
user_data_store_fpath = "public_user_data.csv" 

SAFETY_CHECK = True

CRED_FPATH = "auth.json"
MODEL_NAME_PATH_MAP = {
    "Oil Painting": "models/oilPaintingV10.safetensors",
    "Watercolor": "models/colorwater-v4.safetensors", 
    "MoXin (traditional Chinese Painting)": "models/MoXin-v1.safetensors",
}

IMAGE_STYLE_CHOICES = ["Oil Painting", "Watercolor", "MoXin (traditional Chinese Painting)"]
IMAGE_SIZE_CHOICES = [512, 1024]









