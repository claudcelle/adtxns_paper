import json
import pandas as pd
import os

def load_metadata(simulation_folder_path):
    
    # simulation_folder_path = '/home/claudio/tesi/sarafu/final_simulations/03_fit_actattr_change_s_param'

    files = os.listdir(simulation_folder_path)

    # Identify JSON and Parquet files
    json_files = [f for f in files if f.endswith('_metadata.json')]
    parquet_files = [f for f in files if f.endswith('.parquet')]

    # Pair Parquet and JSON files
    paired_files = []
    for parquet in parquet_files:
        identifier = parquet.split('.parquet')[0]
        json_file = f"{identifier}_metadata.json"
        if json_file in json_files:
            paired_files.append((parquet, json_file))

    # Extract metadata from paired JSON files
    metadata_list = [
        json.load(open(os.path.join(simulation_folder_path, pair[1]))) for pair in paired_files
    ]


    METADATA_DF = pd.DataFrame(metadata_list)

    metadata_df = METADATA_DF.copy(deep=True)
    # metadata_df = metadata_df[metadata_df['T'] != 2]

    # Ensure all columns contain hashable types (e.g., strings for lists)
    metadata_df = metadata_df.map(lambda x: tuple(x) if isinstance(x, list) else x)
    return metadata_df
# metadata_df = metadata_df.set_index('s').sort_index()
