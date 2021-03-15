import gensim
import numpy as np
import pickle
import os

#files = ['HTJ','baseplate','collar','major_capsid','minor_capsid','major_tail','minor_tail','portal','shaft','tail_fiber']
files = ['portal','shaft','tail_fiber']
model = pickle.load(open('d2v_model1.p','rb'))

import model_utils as mutils
for file in files:
    f_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"fasta",file + "_all_clustered.fasta")
    out_path = os.path.join('phage_vecs',file + '_vecs.p')
    mutils.get_vecs(f_path,model,3,out_path)

print('Done')



