import socket
import threading
import json
import random
import string
import smtplib
import pickle
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import time
import base64
from email.message import EmailMessage

from config import CRED_FPATH, NUM_LISTEN

# Configure your email settings for Gmail
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USER = 'guanslabforren@gmail.com'  # Replace with your Gmail address
SMTP_PASSWORD = 'zqtt fpen klfa qzbo'  # Replace with your Gmail password or app-specific password


# Create a lock for synchronizing access to the JSON file
file_lock = threading.Lock()

# Duration for entry validity (1 week)
VALIDITY_DURATION = timedelta(weeks=1)

def generate_password(length=10):
    characters = string.ascii_letters + string.digits + "!&?@"
    return ''.join(random.choice(characters) for _ in range(length))

def send_email(to_email, password, body=None):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = to_email
    msg['Subject'] = 'SD Web AI Tool: Account Registration'
    
    if body is None:
        body = f"Thank you for registering. \n\nYour password is: {password} "\
                "\n\nPlease keep it safe."
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        text = msg.as_string()
        gmail_server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        gmail_server.starttls()
        gmail_server.login(SMTP_USER, SMTP_PASSWORD)
        gmail_server.sendmail(SMTP_USER, to_email, text)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email to {to_email}. Error: {str(e)}")

def cleanup_expired_entries():
    while True:
        time.sleep(86400)  # Run cleanup once a day
        with file_lock:
            try:
                print("-------- Clean up expired entries -------")
                with open(CRED_FPATH, 'r+') as file:
                    try:
                        users = json.load(file)
                    except json.JSONDecodeError:
                        users = {}
                    current_time = datetime.now()
                    users = {email: details for email, details in users.items() if datetime.fromisoformat(details['timestamp']) + VALIDITY_DURATION > current_time}
                    file.seek(0)
                    json.dump(users, file, indent=4)
                    file.truncate()
            except FileNotFoundError:
                continue

def handle_client(client_socket):
    data = client_socket.recv(1024)
    if data is None:
        return
    data = pickle.loads(data)
    email = data['register_email']
    forget = data['forget']
    password = generate_password()
    timestamp = datetime.now().isoformat()
    body = None
    
    if forget:
        with file_lock:
            try:
                with open(CRED_FPATH, 'r+') as file:
                    users = json.load(file)
                    password = users[email]['password']
                    body=f"Please keep the following password save.\n\n{password}"
            except Exception as e:
                print("Error: ------------")
                print(e)
                client_socket.close()
                return
    else: # register
        with file_lock:
            try:
                with open(CRED_FPATH, 'r+') as file:
                    try:
                        users = json.load(file)
                    except json.JSONDecodeError:
                        users = {}
                    users[email] = {'password': password, 'timestamp': timestamp}
                    file.seek(0)
                    json.dump(users, file, indent=4)
                    file.truncate()
            except FileNotFoundError:
                with open(CRED_FPATH, 'w') as file:
                    users = {email: {'password': password, 'timestamp': timestamp}}
                    json.dump(users, file, indent=4)
    
    send_email(email, password, body)
    
    client_socket.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 65431))
    server.listen(NUM_LISTEN)
    print("Server started and listening on port 665432")

    cleanup_thread = threading.Thread(target=cleanup_expired_entries, daemon=True)
    cleanup_thread.start()

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == "__main__":
    start_server()
