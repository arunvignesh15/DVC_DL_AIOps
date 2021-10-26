import yaml
import os
import json
import logging

def read_yaml(path_to_yaml: str):
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"YAML file: {path_to_yaml} loaded Succesfully.")
    return(content)

def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True)
        #print(f"Directory is created at {dir_path}")
        logging.info(f"Directory is created at {dir_path}")
    
def save_local_df(data,data_path,index_status=False):
    data.to_csv(data_path,index=index_status)
    #print(f"Data is saved at {data_path}")
    logging.info(f"Data is saved at {data_path}")

def save_reports(report: dict,report_path: str,indentation=4):
    with open(report_path,"w") as f:
        json.dump(report,f,indent=indentation)
    #print(f" Reports are saved at {report_path}")
    logging.info(f" Reports are saved at {report_path}")