import socket
import json
import struct
import threading

from config import CRED_FPATH

# Dummy function to authenticate users
def authenticate(username, password):
    with open(CRED_FPATH, encoding='utf-8') as f:
        cred = json.load(f)
    if username not in cred or cred[username] != password:
        return False
    return True

# Dummy function to generate an image based on user data
def generate_image(user_data):
    return
    # Replace this with actual image generation logic
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (1024, 1024), color='white')
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), f"User Data: {user_data}", fill='black')
    return image

def handle_client(conn, address):
    print(f"Connection from: {address}")

    try:
        
        recev_info = conn.recv(1024).decode()
        recev_info = json.loads(recev_info)
        
        if 'username' in recev_info and 'password' in recev_info:
            auth_res = authenticate(recev_info.get('username'), recev_info.get('password'))
            if not auth_res:
                response = {'auth_status': 'failure'}
                conn.sendall(json.dumps(response).encode())
                return
            else:
                response = {'auth_status': 'success'}
                conn.sendall(json.dumps(response).encode())
    
        if 'image_request' in recev_info and recev_info['image_request']:
            print("About to generate image...")
            generate_image()
            return
        

        # # Receive user data
        # user_data = conn.recv(1024).decode()
        # user_data_dict = json.loads(user_data)
        # print(f"Received user data: {user_data_dict}")

        # # Generate image based on user data
        # image = generate_image(user_data_dict)
        # image_bytes = io.BytesIO()
        # image.save(image_bytes, format='JPEG')
        # image_data = image_bytes.getvalue()
        # image_size = len(image_data)

        # # Send the size of the image (4 bytes) and the boolean (1 byte)
        # boolean_value = True
        # conn.sendall(struct.pack('>I?', image_size, boolean_value))

        # # Send the actual image data
        # conn.sendall(image_data)
    except Exception as e:
        print(f"Error handling client {address}: {e}")
    finally:
        conn.close()
        print(f"Connection closed for {address}")

def server_program():
    host = "127.0.1.1" 
    port = 4242

    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(1)  # Allow up to 5 connections in the queue
    print(f"Server listening on port {port}")

    while True:
        conn, address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, address))
        client_thread.start()

if __name__ == '__main__':
    server_program()
