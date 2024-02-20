"""
Designs padlock and primer probes for SNAIL in situ amplification
based on input geneID, hybridization sequences, constant and variable 
regions. 
"""

import pandas as pd
import random

from src.path_handler import path_checker

def create_geneid(length, existing):
    '''
    Creates a random geneID sequence and checks if this
    sequence already exists in the provided list of existing
    sequences
    '''
    geneid = ''
    #Checks if combinatorial space is not exhausted
    if len(existing) < (4**length - 4):
        exists = False
        #starts a while loop that is only finished if a new seq is created
        while exists == False:
            #creates a random seq of length length
            geneid = base_randomizer(length)
            if len(set(geneid)) == 1:
                continue
            if not geneid in existing:
                existing.append(geneid)
                #if seq does not exist yet while loop is broken
                exists = True
    else:
        raise Exception(f"The combinatorial space of {4**length} geneID combinations is exhausted")

    return geneid, existing


def base_randomizer(length = 1):
    '''
    Returns a random base sequence with defined length
    '''
    bases = ['A', 'C', 'G', 'T']
    seq = ''
    for i in range(length):
        seq = seq + random.choice(bases)
    return seq


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


def probe_designer(genes_path, probedb_path, output_path, geneids_path = None, middle_const = "AATTATTAC", 
                   end_const_padlock = "CATACACTAAAGATA", end_const_primer = "TATCTT", create_geneids= False, probe_max = 4):
    """
    Creates for a given list of genes primer and padlock seqs
    based on hybridization seqs provided by a database and
    several variable and constant linkers. Can create random
    geneIDs excluding already existing and homopolymers.

    Input: 
        - genes_path: Path to CSV file with list of genes to be used for probe design. 
        Expects the following format with no header:
        gene1
        gene2
        ...
        - probedb_path: Path to CSV file that contains the database with calculated
        target sequences within the transcriptome. While format is not fully specified,
        it must have a header row and one column with name "primer" for the primer target
        sequences and one column named "padlock" for the padlock target sequence
        ...     primer   ...   padlock ...
                pr_seq1        pa_seq1
                pr_seq2        pa_seq2
                ...            ...
        - output_file: Path to desired location and name of the output file, e.g. path/to/probe_df.csv
        - geneids_path: Path to a CSV file that provides one geneID for each gene. Either these geneIDs
        will be used to create the probes, OR if create_geneids = True, it will avoid these geneIDs as
        they will be considered as already existing. If None is provided and create_geneids not True,
        an error will be thrown. While format is not fully specified, it must have a header row and one 
        column with name "gene" for the gene symbol and one column named "geneid" for the geneID:
        ...     gene   ...   geneID ...
                gene1        geneID1
                gene2        geneID2
                ...            ...
        Default: None.
        - middle_const: Sequence of the middle constant in the padlock probe. Default: AATTATTAC (Shi
        et al.)
        - end_const_padlock: Sequence of the end constant in the padlock probe. Default: CATACACTAAAGATA
        (Shi et al.)
        - end_const_primer: Sequence of the end constant in the primer probe. Default: TATCTT (Shi et al.)
        - create_geneids: Can either be set to False, then the gene:geneID combinations need to be 
        provided with the geneids_path parameter. Or can be the length of the random, non homopolymeric
        geneIDs to be created. Cannot be set to True (writing a number is equivalent to setting this parameter
        to True). If create_geneids = int and a gene:geneID assignment given with geneids_path paramter, then
        those geneIDs will be excluded from the creation of random geneIDs.
        - probe_max: Maximum number of probes to design per gene. Default: 4.
    
    Return:
        - A CSV file with genes, geneIDs, probeIDs and probe sequences
    """
    #Test paths
    genes_path = path_checker(genes_path, directory=False)
    if geneids_path:
        geneids_path = path_checker(geneids_path, directory=False)
    probedb_path = path_checker(probedb_path, directory=False)

    geneids = []

    #Extract genes
    genes = pd.read_csv(genes_path, names = ["genes"], header=None)["genes"].tolist()

    #Load the probe database
    probedb = pd.read_csv(probedb_path, header=0)

    #Extracts existing geneIDs
    if geneids_path:
        geneids_df = pd.read_csv(geneids_path)
        geneids_dict = dict(zip(geneids_df.genes, geneids_df.geneids))
        geneids = list(geneids_dict.values())
    elif not create_geneids:
        #If no existing geneids given and create_geneids = False, raises exception
        raise Exception("GeneIDs either need to be provided or the create_geneids parameter set to the"
                        "length of the geneID")
    
    #Extracts a list of existing genes in the target seq database
    dbgenes = probedb["Gene"].values
    #Prepares a dictionary for creating complementary sequences
    comp_dict = {"A" : "T", "T" : "A", "C" : "G", "G" : "C"} 
    #Prepares the final dataframe to safe the created probe sequences
    final_df = pd.DataFrame(columns = ["gene", "geneID", "padlockID", "padlock_seq", "primerID", "primer_seq"])

    #Loops through all genes of interest
    for gene in genes:
        if not gene in dbgenes:
            #If the gene is not in the target database it will be skipped
            print(f"Gene {gene} is not in at least one of the databases and was skipped")
        else:
            if not gene in geneids_dict.keys() and not create_geneids:
                #if a gene does not have a provided geneID and create_geneid = False, gene will
                #skipped
                print(f"Gene {gene} is not in at least one of the databases and was skipped")
            else:
                #creates the start variable with format AXXXTA
                start_var = "A" + base_randomizer(3) + "TA"
                #creates the middle variable as the reverse complementary seq of start_var
                middle_var = ''.join(comp_dict[letter] for letter in start_var[::-1])
                if create_geneids:
                    #creates a random geneIDs (not homopolymer) that does not exist in geneids list
                    geneid, geneids = create_geneid(create_geneids, geneids)
                else:
                    #extract the geneID assigned for gene from geneids dictionary
                    geneid = geneids_dict[gene]
                #Starts a counter to control the number of probes designed per gene
                counter = 0
                    #Loops through all the genes from the target database, this is necessary since 
                    #a single gene can have several target seq
                for i in range(len(dbgenes)):
                    #Check if the max number of probes for the given gene is already reached
                    if counter <= (probe_max-1):
                        #if it finds a gene that is in the list of genes of interest
                        if dbgenes[i] == gene:
                            #builds full primer and padlock probe sequences
                            full_primer = primer_builder(probedb.iloc[i]['primer'], middle_var, end_const_primer)
                            full_padlock = padlock_builder(probedb.iloc[i]['padlock'], start_var, middle_const, geneid, end_const_padlock)
                            #builds the padlock and primer ID
                            padlockID = gene + "_0" + str(counter)
                            primerID = gene + "_1" + str(counter)
                            #saves everything in the dataframe
                            final_df.loc[len(final_df)] = [gene, geneid, padlockID, full_padlock, primerID, full_primer]
                            counter += 1
                        else:
                            break
    #outputs datafram to desired output location
    final_df.to_csv(output_path)
    return final_df
    



