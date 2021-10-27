import tensorflow as tf
import os
from src.utils.all_utils import get_timestamp
import joblib
import logging



def create_and_save_tensorboard_callback(callbacks_dir,tensorboard_log_dir):
    unique_name = get_timestamp("tb_logs")

    tb_running_log_dir = os.path.join(tensorboard_log_dir,unique_name)
    tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    tb_callbacks_filepath = os.path.join(callbacks_dir,"tensorboardcb.cb")
    joblib.dump(tensorboard_callback,tb_callbacks_filepath)
    logging.info(f"Tensorboard callback is being saved at {tb_callbacks_filepath}")

def create_and_save_checkpoint_callback(callbacks_dir,checkpoint_dir):
    checkpoint_file_path =os.path(checkpoint_dir,"ckpt_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_path,
                                                            save_best_only=True
                                                            )
    ckpt_callbacks_filepath = os.path.join(callbacks_dir,"checkpoint.cb")
    joblib.dump(checkpoint_callback,ckpt_callbacks_filepath)
    logging.info(f"Checkpoint callback is being saved at {ckpt_callbacks_filepath}")                            
