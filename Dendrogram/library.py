### dictionary of sequences and structures
### TODO: change to read in dictionaries from text file
from Library_Sequences import *

def get_sequence(sequence):
    for s,v in sequences.items():
        if sequence in s:
            return v

def get_seqlen(sequence):
    for s,v in sequences.items():
        if sequence in s:
            return len(v)

def get_free_structure(sequence):
    for s,v in free_structures.items():
        if sequence in s:
            return v
