#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pickle
import gensim
from Bio.Seq import Seq
from Bio import SeqIO 
import os
import itertools
import torch
import sys
import csv
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# In[3]:


d2v_model = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath("__file__")), "d2v_model1.p"),'rb'))


# In[52]:




def get_vecs(fasta_path,model,kmer_size,embed_dim=250):
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
        vectors.append((vec,rec))
        
        
    print('Number of sequences converted to vectors:',len(vectors))
    return vectors

def get_phanns_input(fasta_list,d2vmodel):
#     d2vmodel = pickle.load(open('d2v_model1.p','rb'))
    AA=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    SC=["1","2","3","4","5","6","7"]
    tri_pep = [''.join(i) for i in itertools.product(AA, repeat = 3)]
    tetra_sc = [''.join(i) for i in itertools.product(SC, repeat = 4)]
    prot_class = 0
    myseq="AILMVNQSTGPCHKRDEFWY"
    trantab2=myseq.maketrans("AILMVNQSTGPCHKRDEFWY","11111222233455566777")
    kmer_size=3
    this_prot=0
    vectors = []
    classes = []
    for file in fasta_list:
        print('####################' + file)
#         file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"fasta",file + "_all_clustered.fasta")
        for record in SeqIO.parse(file, "fasta"):
            ll=len(record.seq)
            seqq=record.seq.__str__().upper()
            seqqq=seqq.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')
            X = ProteinAnalysis(seqqq)
            tt= [X.isoelectric_point(), X.instability_index(),ll,X.aromaticity(),
                X.molar_extinction_coefficient()[0],X.molar_extinction_coefficient()[1],
                X.gravy(),X.molecular_weight()]
            tt_n = np.asarray(tt,dtype=np.float)
            myseq=seqq.translate(trantab2)
            
            #count tripeptides
            tri_pep_count=[seqq.count(i)/(ll-2) for i in tri_pep]
            tri_pep_count_n = np.asarray(tri_pep_count,dtype=np.float)
            
            #count tetra side chains
            tetra_sc_count=[myseq.count(i)/(ll-3) for i in tetra_sc]
            tetra_sc_count_n = np.asarray(tetra_sc_count,dtype=np.float)
            
            #get embedding vector
            vec = d2vmodel.infer_vector([seqqq[k:k+kmer_size] for k in range(0,len(seqqq),kmer_size)])
            for s in range(1,kmer_size):
                vec = vec+d2vmodel.infer_vector([seqqq[k:k+kmer_size] for k in range(s,len(seqqq),kmer_size)])
            vec = vec/kmer_size
            
            cat_n=np.concatenate((tri_pep_count_n,tetra_sc_count_n,tt_n,vec))
            vectors.append((cat_n,record))
            
            
            this_prot+=1
            if (this_prot%500==0):
                print("processing sequence # " + str(this_prot),end="\r")
        prot_class+=1
        this_prot=0
    return vectors
# vecs = []
# for i in range(1,len(sys.argv)):
#     vecs.extend(get_vecs(sys.argv[i],d2v_model,3))

# FILENAME = 'Pt66_Ecoli.faa'
# vecs = get_vecs(FILENAME,d2v_model,3)


# In[53]:


# d2v_model = pickle.load(open('d2v_model1.p','rb'))


# In[82]:


FILENAME= 'Pt66_Ecoli.faa'
vecs = get_phanns_input([FILENAME],d2v_model)


# In[83]:


import torch.nn as nn
from torch.utils.data import TensorDataset
from torch import optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PATH = './phann_model.pth'
embedding_size = 250
num_classes = 11

scaler = pickle.load(open('fitted_scaler.p','rb'))
# vectors = scaler.transform(vecs)

# layers = [nn.Linear(embedding_size,200),
#           nn.ReLU(),
#           nn.Dropout(0.3),
#           nn.Linear(200,200),
#           nn.ReLU(),
#           nn.Linear(200,num_classes)]

# model = nn.Sequential(*layers)

# model.load_state_dict(torch.load(PATH),strict=False)

model = torch.load(PATH)
model.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(type(model))


# In[84]:


# vectors = torch.tensor(vectors).to(device)


# In[85]:


classes =  ['major_capsid','minor_capsid','baseplate','major_tail','minor_tail',
         'portal','tail_fiber','shaft','collar','HTJ','others']
results = []
model.eval()
with torch.no_grad():
    f = True
    for orf in vecs:
        vec = orf[0]
        vec = np.reshape(vec,(1,vec.size))
        vec = scaler.transform(vec)
        vec = torch.tensor(vec)
        
        out = model(vec.to(device).float())
        soft = nn.Softmax(dim=1)
        full=soft(out)
        top = torch.argmax(full)
        pred_class = classes[top.tolist()]
        scores = full.tolist()
#         results.append([orf[1].name,orf[1].description,pred_class,float(score),str(orf[1].seq)])
        
        row = [orf[1].name,orf[1].description,pred_class,str(orf[1].seq)]
        row.extend(scores[0])
        
        if f:
            f=False
            print(len(row))
        results.append(row)
df=pd.DataFrame(results,columns = ['Name','Description','Prediction','Sequence',
                                   'major_capsid','minor_capsid','baseplate','major_tail','minor_tail',
                                     'portal','tail_fiber','shaft','collar','HTJ','others'])


# In[86]:


df.to_csv('Pt66_Ecoli_ph_w_embed.csv')


# In[94]:


orf = vecs[0]
orf = orf[1]
orf.id


# In[ ]:




