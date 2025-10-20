import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import powerlaw as pwl
import pycop
import seaborn as sns
import openturns as ot
import openturns.viewer as viewer
from scipy.stats import entropy
import scipy.stats as st

def balance_constructor_v2(txdf):
    """
    Constructs a balance record DataFrame from transaction data with a different column structure.

    Parameters:
    txdf (DataFrame): Input DataFrame containing at least:
                      ['timestamp', 'source', 'target', 'weight', 'source_bal', 'target_bal', 'date', 'frac_out', 'frac_in']

    Returns:
    DataFrame: A DataFrame with columns ['timestamp', 'date', 'crid', 'balance', 'weight', 'frac'] tracking balance changes.
    """
    # Ensure input has the required columns
    required_cols = ['timestamp', 'source', 'target', 'weight', 'source_bal', 'target_bal', 'date', 'frac_out', 'frac_in']
    if not all(col in txdf.columns for col in required_cols):
        raise ValueError(f"Missing required columns in input DataFrame: {set(required_cols) - set(txdf.columns)}")

    # Construct balance records for source and target
    balances = pd.DataFrame({
        'timestamp': txdf['timestamp'].repeat(2).values,  # Repeat each transaction twice for source and target
        'date': txdf['date'].repeat(2).values,  # Date column
        'crid': pd.concat([txdf['source'], txdf['target']], ignore_index=True),  # User identifiers
        'balance': pd.concat([txdf['source_bal'], txdf['target_bal']], ignore_index=True),  # Balances
        'weight': txdf['weight'].repeat(2).values,
        'frac': pd.concat([txdf['frac_out'], txdf['frac_in']], ignore_index=True),  # Fractional values
    })

    return balances


def balance_constructor(txdf):
    """
    Constructs a balance record DataFrame from transaction data.

    Parameters:
    txdf (DataFrame): Input DataFrame containing at least:
                      ['date', 'source', 'target', 'source_bal_post', 'target_bal_post']

    Returns:
    DataFrame: A DataFrame with columns ['date', 'crid', 'balance'] tracking balance changes.
    """
    # Ensure input has the required columns
    required_cols = ['date', 'source', 'target', 'source_bal_post', 'target_bal_post','weight']
    if not all(col in txdf.columns for col in required_cols):
        raise ValueError(f"Missing required columns in input DataFrame: {set(required_cols) - set(txdf.columns)}")

    # Construct balance records for source and target
    balances = pd.DataFrame({
        'date': txdf['date'].repeat(2).values,  # Repeat each transaction twice for source and target
        'crid': pd.concat([txdf['source'], txdf['target']], ignore_index=True),  # User identifiers
        'balance': pd.concat([txdf['source_bal_post'], txdf['target_bal_post']], ignore_index=True),  # Balances
        'type' : txdf['type'].repeat(2).values,
        'tx_id': txdf['id'].repeat(2).values,
        'weight': txdf.weight.repeat(2).values,
        # 'period': txdf['period'].repeat(2).values
    })

    return balances




def agents_constructor(transaction_dataframe,users_dataframe=None, standard=True, end=None, begin=None, how='outer'):
    if users_dataframe is None:
        try:
            users_dataframe = USERS
        except NameError:
            raise ValueError("USERS is not defined in this environment. Please provide a value for 'users_dataframe'.")
    # Check if end and begin are provided, otherwise use END and START if they exist
    if end is None:
        try:
            end = END
        except NameError:
            raise ValueError("END is not defined in this environment. Please provide a value for 'end'.")
    
    if begin is None:
        try:
            begin = START
        except NameError:
            raise ValueError("START is not defined in this environment. Please provide a value for 'begin'.")

    # Aggregation for sources
    sources = transaction_dataframe.groupby("source").agg({
                                    'weight':['count','sum','mean','median','max','min'],
                                    'frac_out':['mean','median','max','min'], # I added this line
                                    'target':'nunique',
                                    'date':'first'})
    
    # Correct column names after aggregation
    sources.columns = [
        "total_txns_out", "total_exp", "mean_exp", "median_exp", "max_exp", "min_exp",  # weight columns
        "mean_frac_out", "median_frac_out", "max_frac_out", "min_frac_out",            # frac_out columns
        "num_targets", "first_txns_out"                                                # target and date columns
    ]
    sources.reset_index(inplace=True)
    
    sources['opening'] = sources['source'].map(users_dataframe.set_index('crid')['start'])
    sources['frac_from_1st_out'] = abs(end - (sources.first_txns_out)) / abs(end - begin)
    sources['frac_from_op'] = abs(end - (sources.opening.where(sources.opening > begin, begin))) / abs(end - begin)
    sources['eff_txns_out'] = sources.total_txns_out / sources.frac_from_op
    sources['eff_outdegree'] = sources.num_targets / sources.frac_from_op

    # Aggregation for targets
    targets = transaction_dataframe.groupby("target").agg({
                                    'weight': ['count','sum','mean','median','max','min'],
                                    'frac_in': ['mean','median','max','min'], # I added this line
                                    'source':'nunique',
                                    'date':'first'})
    
    # Correct column names after aggregation
    targets.columns = [
        "total_txns_in", "total_earn", "mean_earn", "median_earn", "max_earn", "min_earn",  # weight columns
        "mean_frac_in", "median_frac_in", "max_frac_in", "min_frac_in",                    # frac_in columns
        "num_sources", "first_txns_in"                                                    # source and date columns
    ]
    targets.reset_index(inplace=True)

    targets['opening'] = targets['target'].map(users_dataframe.set_index('crid')['start'])
    targets['frac_from_1st_in'] = abs(end - (targets.first_txns_in)) / abs(end - begin)
    targets['frac_from_op'] = abs(end - (targets.opening.where(targets.opening > begin, begin))) / abs(end - begin)
    targets['eff_txns_in'] = targets.total_txns_in / targets.frac_from_op
    targets['eff_indegree'] = targets.num_sources / targets.frac_from_op

    targets.rename(columns={'target': 'crid'}, inplace=True)
    sources.rename(columns={'source': 'crid'}, inplace=True)

    agents = pd.merge(targets, sources, on=['crid','opening','frac_from_op'], how=how)
    
    if how == 'outer':
        agents = agents.fillna(0)
    
    if standard:
        agents = agents.rename(columns={
            'total_txns_in':'attractiveness',
            'total_txns_out':'activity',
            'total_earn':'vol_in',
            'total_exp':'vol_out',
            'num_sources':'in_deg',
            'num_targets':'out_deg',
            'eff_txns_in': 'eff_attr',
            'eff_txns_out': 'eff_act',
        })
        agents = agents[['crid', 'frac_from_op',
                         'attractiveness','activity', 
                         'vol_in','vol_out',
                         'in_deg','out_deg',
                         'eff_attr', 'eff_act',
                         'eff_indegree', 'eff_outdegree',
                         'first_txns_in', 'first_txns_out',
                         'frac_from_1st_in','frac_from_1st_out',
                         'mean_earn', 'mean_exp',
                         'median_earn', 'median_exp',
                         'max_earn', 'max_exp',
                         'min_earn', 'min_exp'
                    ]]
    else:
        agents = agents[['crid', 'opening','frac_from_op',
                         'total_txns_in','total_txns_out', 
                         'total_earn','total_exp',
                         'num_sources','num_targets',
                         'eff_txns_in', 'eff_txns_out',
                         'eff_indegree', 'eff_outdegree',
                         'first_txns_in', 'first_txns_out',
                         'frac_from_1st_in','frac_from_1st_out',
                         'mean_earn', 'mean_exp',
                         'median_earn', 'median_exp', 
                         'max_earn', 'max_exp',
                         'min_earn', 'min_exp'
                    ]]
    agents = agents.merge(users_dataframe,how='left',on='crid')

    return agents



def basic_agents(transaction_dataframe,user_df = None,tx_type = 'STANDARD',how='outer'):
    # if user_df is None:
    #     try:
    #         user_df = USERS.copy(deep=True)
    #     except exception as e:
    #         print(e)

    sources = transaction_dataframe.groupby("source").agg({
                                    'weight':['count','sum','mean','median','max','min'],
                                    'frac_out':['mean','median','max','min'], # I added this line
                                    'target':'nunique',
                                    'date':'first'})
    
    # Correct column names after aggregation
    sources.columns = [
        "total_txns_out", "total_exp", "mean_exp", "median_exp", "max_exp", "min_exp",  # weight columns
        "mean_frac_out", "median_frac_out", "max_frac_out", "min_frac_out",            # frac_out columns
        "num_targets", "first_txns_out"                                                # target and date columns
    ]
    sources.reset_index(inplace=True)

    targets = transaction_dataframe.groupby("target").agg({
                                    'weight': ['count','sum','mean','median','max','min'],
                                    'frac_in': ['mean','median','max','min'], # I added this line
                                    'source':'nunique',
                                    'date':'first'})
    
    # Correct column names after aggregation
    targets.columns = [
        "total_txns_in", "total_earn", "mean_earn", "median_earn", "max_earn", "min_earn",  # weight columns
        "mean_frac_in", "median_frac_in", "max_frac_in", "min_frac_in",                    # frac_in columns
        "num_sources", "first_txns_in"                                                    # source and date columns
    ]
    targets.reset_index(inplace=True)

    targets.rename(columns={'target': 'crid'}, inplace=True)
    sources.rename(columns={'source': 'crid'}, inplace=True)

    agents = pd.merge(targets, sources, on=['crid'], how=how)
    # agents['activity']
    
    if how == 'outer':
        agents = agents.fillna(0)
    agents = agents.rename(columns={
            'total_txns_in':'tx_in',
            'total_txns_out':'tx_out',
            'total_earn':'vol_in',
            'total_exp':'vol_out',
            'num_sources':'in_deg',
            'num_targets':'out_deg',
            
        })
    
    # agents = agents[['crid', 'frac_from_op',
    #                     'attractiveness','activity', 
    #                     'vol_in','vol_out',
    #                     'in_deg','out_deg',
    #                     'eff_attr', 'eff_act',
    #                     'eff_indegree', 'eff_outdegree',
    #                     'first_txns_in', 'first_txns_out',
    #                     'frac_from_1st_in','frac_from_1st_out',
    #                     'mean_earn', 'mean_exp',
    #                     'median_earn', 'median_exp',
    #                     'max_earn', 'max_exp',
    #                     'min_earn', 'min_exp'
    #             ]]
    return agents



def find_key(b,s,test=False):
    for key, simu in sim_results.items():
        if test:
            print(f"{key} : b = {simu['b']}, s = {simu['s']}")

        if simu['b'] == b and simu['s'] == s:
            key_to_return = key

    return key_to_return

def round_df(df, precision = 3, includes='float'):
    df[df.select_dtypes(include=includes).columns] = df.select_dtypes(include='float').apply(
        lambda x: x.round(precision)    )
    return df
