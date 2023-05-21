import pandas as pd
import numpy as np

VOCAB_SIZE = 64 #number of codons

codon_table = {'TCA': 'S', 'AAT': 'N', 'TGG': 'W', 'GAT': 'D', 'GAA': 'E', 'TTC': 'F', 'CCG': 'P',
           'ACT': 'T', 'GGG': 'G', 'ACG': 'T', 'AGA': 'R', 'TTG': 'L', 'GTC': 'V', 'GCA': 'A',
           'TGA': '*', 'CGT': 'R', 'CAC': 'H', 'CTC': 'L', 'CGA': 'R', 'GCT': 'A', 'ATC': 'I',
           'ATA': 'I', 'TTT': 'F', 'TAA': '*', 'GTG': 'V', 'GCC': 'A', 'GAG': 'E', 'CAT': 'H',
           'AAG': 'K', 'AAA': 'K', 'GCG': 'A', 'TCC': 'S', 'GGC': 'G', 'TCT': 'S', 'CCT': 'P',
           'GTA': 'V', 'AGG': 'R', 'CCA': 'P', 'TAT': 'Y', 'ACC': 'T', 'TCG': 'S', 'ATG': 'M',
           'TTA': 'L', 'TGC': 'C', 'GTT': 'V', 'CTT': 'L', 'CAG': 'Q', 'CCC': 'P', 'ATT': 'I',
           'ACA': 'T', 'AAC': 'N', 'GGT': 'G', 'AGC': 'S', 'CGG': 'R', 'TAG': '*', 'CGC': 'R',
           'AGT': 'S', 'CTA': 'L', 'CAA': 'Q', 'CTG': 'L', 'GGA': 'G', 'TGT': 'C', 'TAC': 'Y',
           'GAC': 'D'}

#amino acid table that maps amino acids to a list of codons that creates them
amino_acid_table = {'S': ['TCA', 'TCC', 'TCT', 'TCG', 'AGC', 'AGT'], 'N': ['AAT', 'AAC'], 'W': ['TGG'],
          'D': ['GAT', 'GAC'], 'E': ['GAA', 'GAG'], 'F': ['TTC', 'TTT'], 'P': ['CCG', 'CCT', 'CCA', 'CCC'],
          'T': ['ACT', 'ACG', 'ACC', 'ACA'], 'G': ['GGG', 'GGC', 'GGT', 'GGA'],
          'R': ['AGA', 'CGT', 'CGA', 'AGG', 'CGG', 'CGC'], 'L': ['TTG', 'CTC', 'TTA', 'CTT', 'CTA', 'CTG'],
          'V': ['GTC', 'GTG', 'GTA', 'GTT'], 'A': ['GCA', 'GCT', 'GCC', 'GCG'], '*': ['TGA', 'TAA', 'TAG'],
          'H': ['CAC', 'CAT'], 'I': ['ATC', 'ATA', 'ATT'], 'K': ['AAG', 'AAA'], 'Y': ['TAT', 'TAC'],
          'M': ['ATG'], 'C': ['TGC', 'TGT'], 'Q': ['CAG', 'CAA']}

codon_freq_ecoli = {'TTT': 22.38, 'TCT': 8.61, 'TAT': 16.36, 'TGT': 5.19, 'TTC': 16.21,
                      'TCC': 8.81, 'TAC': 12.15, 'TGC': 6.34, 'TTA': 13.83, 'TCA': 7.57,
                      'TAA': 2.03, 'TGA': 1.04, 'TTG': 13.37, 'TCG': 8.79, 'TAG': 0.25,
                      'TGG': 15.21, 'CTT': 11.44, 'CCT': 7.22, 'CAT': 12.84, 'CGT': 20.7,
                      'CTC': 10.92, 'CCC': 5.56, 'CAC': 9.44, 'CGC': 21.48, 'CTA': 3.93,
                      'CCA': 8.44, 'CAA': 15.1, 'CGA': 3.67, 'CTG': 52.1, 'CCG': 22.65,
                      'CAG': 29.21, 'CGG': 5.72, 'ATT': 30.21, 'ACT': 9.02, 'AAT': 18.26,
                      'AGT': 9.08, 'ATC': 24.6, 'ACC': 22.88, 'AAC': 21.47, 'AGC': 15.89,
                      'ATA': 4.88, 'ACA': 7.63, 'AAA': 33.94, 'AGA': 2.43, 'ATG': 27.59,
                      'ACG': 14.47, 'AAG': 10.7, 'AGG': 1.48, 'GTT': 18.39, 'GCT': 15.54,
                      'GAT': 32.43, 'GGT': 24.45, 'GTC': 15.07, 'GCC': 25.45, 'GAC': 19.14,
                      'GGC': 28.65, 'GTA': 10.97, 'GCA': 20.61, 'GAA': 39.55, 'GGA': 8.44,
                      'GTG': 25.9, 'GCG': 32.79, 'GAG': 18.24, 'GGG': 11.29}


def get_codons(seq):
    #return ''.join([seq[s:s+3] for s in range(0, len(seq), 3)])
    return ' '.join([seq[s:s+3] for s in range(0, len(seq), 3)])

def get_codon_list(sequences):
    return np.array([get_codons(seq) for seq in sequences], dtype=object)

def clean(df, col):
    return df[df[col].map(lambda d: len(str(d)) > 9 and len(str(d)) % 3 == 0 and set(str(d)) == set('ACTGU'))]

def add_codons_to_df(df, col):
    df = clean(df, col)
    df['codons_cleaned'] = get_codon_list(df[col])
    return df

def norm(list):
    range_value = max(list) - min(list)
    range_value = range_value + range_value/50
    min_value = min(list) - range_value/100

    # subtract the minimum value and divide by the range
    for index, item in enumerate(list):
        list[index] = (item - min_value) / range_value

    return list
