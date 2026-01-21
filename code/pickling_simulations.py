import gc
from tqdm import tqdm
import os 
from pathlib import Path
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from itertools import cycle
import pickle
import json
from typing import Union,List

from adtxns_utils import *
from stats_utils import *
from preprocessing_utils import *
import sys 

def main(batch):
    END = pd.Timestamp('2021-06-15 05:00:30.887568') #*2021-06-15 05:00:30.887568
    START = pd.Timestamp('2020-01-25 19:13:17.731529') #*2020-01-25 19:13:17.731529
    FIRST_RENEWAL_TIME = 47318.40424 #*seconds from the first activation event in the reference dataset 
    NUM_PERIODS = 20 #*free parameter, number of subperiods to divide the transaction record (both sarafu or synthetic data)

    batch = batch
    simulations_dir = Path("/home/claudio/tesi/sarafu/final_simulations/")
    in_dir = simulations_dir / batch

    pickles_dir = Path("../pickled_data/")
    out_dir = pickles_dir / batch
    os.makedirs(out_dir, exist_ok=True)

    metadata_df = load_metadata(in_dir)

    for n in range(len(metadata_df)//5):
        for _, current_row in tqdm(metadata_df.iloc[(n*5):((n+1)*5),:].iterrows()):
            sim_dict = process_single_simulation(
                row=current_row,
                folder_path=in_dir,
                first_renewal_time=FIRST_RENEWAL_TIME,
                start=START,
                compute_iet_flag=True,
            )

            out_path = os.path.join(out_dir, f"{current_row.simulation_id}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(sim_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            del sim_dict
            gc.collect()



if __name__ == "__main__":
    if len(sys.argv) > 1:
        batch = sys.argv[1]
        main(batch)
    else:
        raise Exception("Please provide a batch name as an argument.")
    
