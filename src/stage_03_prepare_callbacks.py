from src.utils.all_utils import read_yaml,create_directory
from src.utils.models import get_VGG16_model , prepare_model
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import logging
import io

logging_str ="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s]"
log_dir="logs"
#create_directory([log_dir])
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode='a')

def prepare_callbacks(config_path,params_path):
    pass

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info(">>>>>> Stage three Started")
        prepare_callbacks(config_path=parsed_args.config,params_path=parsed_args.params)
        logging.info("Stage three completed..! Callbacks are prepared and saved as binary.>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e

