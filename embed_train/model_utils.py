import gensim
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO 
import os
import pickle


def get_vecs(fasta_path,model,kmer_size,out_path,embed_dim=250):
    records = list(SeqIO.parse(fasta_path,"fasta"))
    vectors = []
    print('processing ', fasta_path)

    for rec in records:
        #format the sequence string by making uppercase...
        #replace X with A, J with L, * with A, Z with E, and B with D
        string = str(rec.seq).upper()
        string = string.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')
        vec = model.infer_vector([string[k:k+kmer_size] for k in range(0,len(string),kmer_size)])
        for s in range(1,kmer_size):
            vec = vec+model.infer_vector([string[k:k+kmer_size] for k in range(s,len(string),kmer_size)])
        vec = vec/kmer_size
        vectors.append(vec)
        print('done')
        
    print('Number of sequences converted to vectors:',len(vectors))
    pickle.dump(vectors,open(out_path,'wb'),protocol=4)
    print(f'File saved to {out_path}.p',out_path)

