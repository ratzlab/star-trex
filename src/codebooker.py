"""
Creates and handles codebooks
"""

from starfish import Codebook
import pandas as pd 
import numpy as np 
import os
from path_handler import path_checker


def onebase_encoder(codebook_np, codekey_dict, rawcode_df, n_rounds):
    """
    Encodes codewords based on a one-base system
    """
    for i in range(len(rawcode_df)):
        for j in range(n_rounds):
            codebook_np[i, j, codekey_dict[rawcode_df["target_id"][i][j]] - 1] = 1
    return codebook_np


def twobases_encoder(codebook_np, codekey_dict, rawcode_df, n_rounds, border_base):
    """
    Encodes codewords based on a two-bases system
    """
    for i in range(len(rawcode_df)):
        for j in range(n_rounds):
            #Extract code snippet
            if j == 0: 
                snippet = border_base + rawcode_df["target_id"][i][j] 
            elif j == n_rounds-1:
                snippet = rawcode_df["target_id"][i][j-1] + border_base
            else:
                snippet = rawcode_df["target_id"][i][j-1] + rawcode_df["target_id"][i][j]

            #Encode snippet
            codebook_np[i, j, codekey_dict[snippet]-1] = 1
    return codebook_np


def store_codebook(codebook, output_path):
    """
    Stores codebook as JSON file
    """
    output_path = path_checker(output_path, exists=False, directory=False, json=True)
    codebook.to_json(str(output_path))


def create_codebook(
        code_path: str,
        key_path: str,
        n_channels: int,
        n_rounds: int,
        output_path=None,
        two_bases_code=False,
        border_base="G"
):
    """
    Returns a starfish Codebook (starfish.core.codebook.codebook) from an input CSV file.
    Optionally, the Codebook can be stored as JSON file.

    Input: 
        - code_path: Path to CSV file assigning each target (gene, cloneID) to one geneID.
        The file mus have the following format [target_name, target_id]:
        cloneID1,AGAAA
        cloneID2,TGATT
        ...
        - key_path: Path to CSV file assigning base/combination of bases to one channel.
        The file must have the following format: [base(s), channel]. For example:
        A,1
        C,2
        G,3
        T,4
        - n_channels: Number of channels
        - n_rounds: Number of rounds
        - output_path (optional): path to output location of JSON file, default: None
        - two_bases_code: Boolean, if True a two-bases key will be used, default: False
        - border_base: Base that borders the target_id, default: G
    """
    #1. Check if provided paths are correct
    code_path = path_checker(code_path, directory=False)
    key_path = path_checker(key_path, directory=False)

    #2. Read input files
    rawcode_df = pd.read_csv(code_path, names=["target_name", "target_id"], header=None)
    codekey_df = pd.read_csv(key_path, names=["base", "channel"], header=None)
    codekey_dict = pd.Series(codekey_df.channel.values,index=codekey_df.base).to_dict()
    
    #3. Create a numpy array from target_ids and code_key
    codebook_np = np.zeros((len(rawcode_df),n_rounds,n_channels))
    if two_bases_code:
        codebook_np = twobases_encoder(codebook_np, codekey_dict, rawcode_df, n_rounds, border_base)
    else:
        codebook_np = onebase_encoder(codebook_np, codekey_dict, rawcode_df, n_rounds)

    #4. Transform into starfish Codebook format
    codebook = Codebook.from_numpy(rawcode_df["target_name"], n_rounds, n_channels, data=codebook_np)
    
    #5. Optional: Store Codebook as JSON file
    if output_path:
        store_codebook(codebook, output_path)

    return codebook


def load_codebook(codebook_path):
    """
    Loads a codebook from a JSON file
    """

    codebook_path = path_checker(codebook_path, directory=False)
    codebook = Codebook.open_json(str(codebook_path))
    return codebook