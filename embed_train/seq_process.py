# %%
import sys, os, itertools
from Bio.Seq import Seq
from Bio import SeqIO
import numpy as np
import pickle

# %%
#euk == eukaryotic, prok == prokaryotic, arch = archael, phage == phage, vir == virus

#phage_prok produces very large file
files = ["phage_euk_dr100",
         "phage_prok_dr100",
         "phage_arch_dr100",
         "vir_euk_dr100",
         "vir_prok_dr100",
         "vir_arch_dr100"]


# %%
#Function for turning raw fasta files into pickle files containing training items for embedding
#Document vector comes from weighted sum of kmer embeddings
#Context window is number of kmers before and after reference kmer
def get_kmer_embed_training(files,kmer_size,context_size):
    print("Starting fasta conversion to embedding training ready files.\n\n")
    
    #Create dictionary for kmers
    alphabet=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    kmer_dict = {}
    for count, kmer in enumerate(itertools.product(alphabet,repeat=kmer_size),1):
        kmer_dict["".join(kmer)] = count
    
    #Iterate through all input files
    for file in files:
        file_in = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"fasta",file + ".fasta")
        
        print("Parsing " + file + ".fasta")
        records = list(SeqIO.parse(file_in,"fasta"))
        
        #string parameter is sequence to be processed
        #train_type is either "cbow" or "skipgram"
        def record_to_context_windows(record):
            out = []
            #format the sequence string by making uppercase...
            #replace X with A, J with L, * with A, Z with E, and B with D
            string = str(record.seq).upper()
            string = string.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')

            #Different non-overlapping windows
            for i in [0,1,2]:
                #Create kmers, map to integer value from dictionary
                kmer_strings = [string[k:k+kmer_size] for k in range(i,len(string),kmer_size)]
                kmer_ints = [kmer_dict.get(ks) for ks in kmer_strings if len(ks)==kmer_size]
                
                if(len(kmer_ints) < context_size*2+1):
                    continue
                for j in range(context_size,len(kmer_ints)-context_size):
                    context = [kmer_ints[g] for g in list(range(j-context_size,j)) + list(range(j+1, j+1+context_size))]
                    target = kmer_ints[j]
                    out.append((context,target,record.id))
            return out
        
        
        #Iterate through every sequence in file
        print("Converting sequences into training data" )
        context_windows = []
        
        s = 0 #seqcount
        for rec in records:
            context_windows.extend(record_to_context_windows(rec))
            sys.stdout.write("\rSequences processed in " + file + ".fasta : %s" % s)
            sys.stdout.flush()
            s+=1
        
        print("\n" + file + ".fasta converted to training data with " + str(len(context_windows)) + " examples.\n")
        
        output_path = os.path.join(os.path.dirname(os.path.abspath("__file__")),"processed_data", file + ".p")
        pickle.dump(context_windows,open(output_path,"wb"),protocol=4)
        print("Training data saved to " + output_path + ".\n\n\n")
        
                    
                
    

# %%
get_kmer_embed_training(files,3,4)

# %%

# %%
