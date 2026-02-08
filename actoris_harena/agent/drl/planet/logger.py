import os
import pandas as pd

def loss_logger(losses_dict, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    losses_df = pd.DataFrame.from_dict(losses_dict)
    losses_df.to_csv(
        os.path.join(save_dir, 'losses.csv'), 
        mode= ('w' if losses_dict['update_step'][0] == 0 else 'a'), 
        header= (True if losses_dict['update_step'][0] == 0 else False)
    )

def eval_logger(eval_dict, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    evalutate_df = pd.DataFrame.from_dict(eval_dict)
    evalutate_df.to_csv(
        os.path.join(save_dir, 'evaluation.csv'), 
        mode= ('w' if eval_dict['update_step'][0] == 0 else 'a'), 
        header= (True if eval_dict['update_step'][0] == 0 else False)
    
    )