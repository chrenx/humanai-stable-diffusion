import socket
import threading
import json
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import NUM_LISTEN

# Configure your email settings
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USER = 'guanslabforren@gmail.com'
SMTP_PASSWORD = 'daydayup2024!'

gmail_server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
gmail_server.starttls()
gmail_server.login(SMTP_USER, SMTP_PASSWORD)

def generate_password(length=10):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

def send_email(to_email, password):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = to_email
    msg['Subject'] = 'SD Web AI Tool: Account Registration'
    
    body = f"Thank you for registering. \nYour password is: {password} \n\n"\
            "Here is the web link: https://guanlab-sdwebui.dcmb.med.umich.edu/"
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        text = msg.as_string()
        gmail_server.sendmail(SMTP_USER, to_email, text)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email to {to_email}. Error: {str(e)}")

def handle_client(client_socket):
    email = client_socket.recv(1024).decode('utf-8')
    password = generate_password()

    try:
        with open('users.json', 'r+') as file:
            try:
                users = json.load(file)
            except json.JSONDecodeError:
                users = {}
            users[email] = password
            file.seek(0)
            json.dump(users, file, indent=4)
            file.truncate()
    except FileNotFoundError:
        with open('users.json', 'w') as file:
            users = {email: password}
            json.dump(users, file, indent=4)
    
    send_email(email, password)
    
    client_socket.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 66432))
    server.listen(NUM_LISTEN)
    print("Server started and listening on port 9999")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == "__main__":
    start_server()
