"""
Designs padlock and primer probes for SNAIL in situ amplification
based on input geneID, hybridization sequences, constant and variable 
regions. 
"""

import pandas as pd
import random

from src.path_handler import path_checker

def base_randomizer():
    '''
    Returns a random base
    '''
    bases = ['A', 'C', 'G', 'T']
    return random.choice(bases)


def primer_builder(primer, middle_var, end_const_primer):
    '''
    Puts together the full primer seq from all elements
    '''
    full_primer = primer + middle_var + end_const_primer
    return full_primer


def padlock_builder(padlock, start_var, middle_const, genenid, end_const_padlock):
    '''
    Puts together the full padlock seq from all elements
    '''
    full_padlock = start_var + padlock + middle_const + genenid + end_const_padlock
    return full_padlock


def probe_designer(genes_path, geneIDs_path, probedb_path, output_path, middle_const = "AATTATTAC", 
                   end_const_padlock = "CATACACTAAAGATA", end_const_primer = "TATCTT"):
    """
    Creates for a given list of genes primer and padlock seqs
    based on hybridization seqs provided by a database and
    several variable and constant linkers

    Input:
    - 
    """
    #Test paths
    genes_path = path_checker(genes_path, directory=False)
    geneIDs_path = path_checker(geneIDs_path, directory=False)
    probedb_path = path_checker(probedb_path, directory=False)

    #Extract genes
    genes = pd.read_csv(genes_path, names = ["genes"], header=None)["genes"].tolist()

    #Load the probe database
    probedb = pd.read_csv(probedb_path, header=0)

    #Extract geneIDs, primer seqs and padlock seqs and saves full primer/padlock seq as csv
    geneids_df = pd.read_csv(geneIDs_path, names = ["genes", "ids"])
    geneids_dict = dict(zip(geneids_df.genes, geneids_df.ids))
    dbgenes = probedb["Gene"].values
    lost_genes = []
    comp_dict = {"A" : "T", "T" : "A", "C" : "G", "G" : "C"} 

    final_df = pd.DataFrame(columns = ["gene", "geneID", "padlockID", "padlock_seq", "primerID", "primer_seq"])

    for gene in genes:
        start_var = "A" + base_randomizer() + base_randomizer() + base_randomizer() + "TA"
        middle_var = ''.join(comp_dict[letter] for letter in start_var[::-1])

        if not gene in geneids_dict.keys() or not gene in dbgenes:
            lost_genes.append(gene)
            print(f"Gene {gene} is not in at least one of the databases and was skipped")
        else:
            geneid = geneids_dict[gene]
            counter = 0
            for i in range(len(dbgenes)):
                if dbgenes[i] == gene:
                    counter += 1
                    full_primer = primer_builder(probedb.iloc[i]['Primer'], middle_var, end_const_primer)
                    full_padlock = padlock_builder(probedb.iloc[i]['Padlock'], start_var, middle_const, geneid, end_const_padlock)
                    padlockID = gene + "_0" + str(counter)
                    primerID = gene + "_1" + str(counter)
                    final_df.loc[len(final_df)] = [gene, geneid, padlockID, full_padlock, primerID, full_primer]
    final_df.to_csv(output_path)
    return final_df

    



