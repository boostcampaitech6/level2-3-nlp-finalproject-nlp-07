from fastapi import Request
from datetime import datetime
import os
from settings import LOGS_DIR

import logging
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_visit(req: Request):
    logging.info("log_visit")
    cli_ip = req.client.host
    cli_ip = cli_ip.replace(".", "_")
    
    today = datetime.today().strftime("%Y%m%d")
    if not os.path.exists(os.path.join(LOGS_DIR, 'visit', today)):
        os.makedirs(os.path.join(LOGS_DIR, 'visit', today))
        
    if os.path.exists(os.path.join(LOGS_DIR, 'visit', today, f"{cli_ip}.txt")):
        return
        
    logging.info(f"new visit - client_ip : {cli_ip}, today : {today}")
    with open(os.path.join(LOGS_DIR, 'visit', today, f"{cli_ip}.txt"), 'a') as f:
        f.write(f"{datetime.now()}, {cli_ip}\n")
        
def log_generate(req: Request, input_logs):
    '''
    input_logs = {
        "input_text": text,
        "translate_text": text,
        "condition": f'emotion : {condition[0]}, tempo : {condition[1]}, genre : {condition[2]}'
    }
    '''
    logging.info("log_generate")
    cli_ip = req.client.host
    cli_ip = cli_ip.replace(".", "_")
    
    today = datetime.today().strftime("%Y%m%d")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(os.path.join(LOGS_DIR, 'generate', today)):
        os.makedirs(os.path.join(LOGS_DIR, 'generate', today))
    
    with open(os.path.join(LOGS_DIR, 'generate', today, f"{cli_ip}_{now}.txt"), 'w') as f:
        f.write(f"client_ip : {cli_ip}\n")
        f.write(f"today : {today}\n")
        for key, value in input_logs.items():
            f.write(f"{key} : {value}\n")

    logging.info(f"new generate - client_ip : {cli_ip}, today : {today}")


def count_total_visit():
    # count the number of  text files in logs/visit
    visit_dir = os.path.join(LOGS_DIR, 'visit')
    total_visit = 0
    for root, dirs, files in os.walk(visit_dir):
        total_visit += len(files)
    
    return total_visit

def count_today_visit():
    # count the number of text files in logs/visit/today
    today = datetime.today().strftime("%Y%m%d")
    today_dir = os.path.join(LOGS_DIR, 'visit', today)
    today_visit = 0
    for root, dirs, files in os.walk(today_dir):
        today_visit += len(files)
    
    return today_visit

def count_total_generate():
    # count the number of text files in logs/generate
    generate_dir = os.path.join(LOGS_DIR, 'generate')
    total_generate = 0
    for root, dirs, files in os.walk(generate_dir):
        total_generate += len(files)
    
    return total_generate

def count_today_generate():
    today = datetime.today().strftime("%Y%m%d")
    today_dir = os.path.join(LOGS_DIR, 'generate', today)
    today_generate = 0
    for root, dirs, files in os.walk(today_dir):
        today_generate += len(files)
        
    return today_generate