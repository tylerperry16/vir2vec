#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO 
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
import pickle
import itertools
import torch


# In[17]:


def get_input(fasta_list):
    d2vmodel = pickle.load(open('d2v_model1.p','rb'))
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
    records = []
    alldata = []
    for grp in range(1,12):
        print('***Group ',grp)
        for file in fasta_list:
            print('####################' + file)
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),
                                     "fasta","expanded",str(grp) + "_" + file + ".fasta")
            for record in SeqIO.parse(file_path, "fasta"):

                ll=len(record.seq)
                seqq=record.seq.__str__().upper()
                seqqq=seqq.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')
                #Extra features
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
                vectors.append(cat_n)
                classes.append(prot_class)
                records.append(record)
                this_prot+=1
                if (this_prot%100==0):
                    print("processing sequence # " + str(this_prot),end="\r")
            prot_class+=1
            this_prot=0
        alldata.append((vectors,classes,records))
        records = []
        classes = []
        vectors = []
    return alldata
            
            


# In[13]:


def count_seqs(file):
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),
                                 "fasta",file + "_all_clustered.fasta")
    count = 0
    for record in SeqIO.parse(file_path, "fasta"):
        count+=1
    print(file, ": ",count)


# In[14]:


count_seqs('others')


# In[ ]:


files = ['major_capsid','minor_capsid','baseplate','major_tail','minor_tail',
         'portal','tail_fiber','shaft','collar','HTJ','other']
alldata = get_input(files)


# In[ ]:


pickle.dump(alldata, open('allfeatures.p','wb'),protocol=pickle.HIGHEST_PROTOCOL)


# In[26]:


print(len(alldata))


# In[46]:


len((alldata[2][0]))


# In[21]:


len(vectors)


# In[22]:


outcls = [10 for x in range(len(outvecs))]


# In[23]:


len(outcls)


# In[25]:


pickle.dump(vectors,open('phannvecs.p','wb'),protocol=pickle.HIGHEST_PROTOCOL)


# In[26]:


classes.extend(outcls)


# In[27]:


pickle.dump(classes,open('phannclasses.p','wb'),protocol=pickle.HIGHEST_PROTOCOL)


# In[29]:


vectors = np.array(vectors)
classes = np.array(classes)


# In[30]:


vectors


# In[4]:


def get_train_test_split(vecs,classes):

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    print('Generating train and test data')
    np.random.seed(1)
    ind = np.arange(len(vecs))
    np.random.shuffle(ind)
    X=vecs[ind]
    train_X=X[0:int(len(X)*.9)]
    test_X=X[int(len(X)*.9):]
    Y=classes[ind]
    train_Y = Y[0:int(len(Y)*.9)]
    test_Y=Y[int(len(Y)*.9):]
    return train_X,train_Y,test_X,test_Y


# In[2]:


vectors = pickle.load(open('phannvecs.p','rb'))
classes = pickle.load(open('phannclasses.p','rb'))


# In[129]:


vectors = np.array(vectors)
classes = np.array(classes)


# In[157]:


train_X,train_Y_index,test_X,test_Y_index = get_train_test_split(vectors,classes)
# train_X,train_Y,test_X,test_Y = map(torch.tensor,(train_X,train_Y,test_X,test_Y))


# In[130]:


print(train_X.shape,train_Y_index.shape,test_X.shape,test_Y_index.shape)


# In[175]:


train_Y_index_t = np.zeros(len(train_Y_index),dtype=int)
test_Y_index_t = np.zeros(len(test_Y_index),dtype=int)
corr_ind=[3,4,1,5,6,7,9,8,2,0,10]
# for i in range(11):
#     train_Y_index_t[train_Y_index==corr_ind[i]]=i
#     test_Y_index_t[test_Y_index==corr_ind[i]]=i
# #     test_Y_index_t = np.where(test_Y_index_t==corr_ind[i],i,test_Y_index_t)
ran = list(range(11))

testvals = np.random.randint(0,11,30)
tvals = testvals
corr_dict = dict(zip(corr_ind,ran))

for i in range(len(testvals)):
    tvals[i] = corr_dict[testvals[i]]
for i in range(len(train_Y_index)):
    train_Y_index_t[i] = corr_dict[train_Y_index[i]]
for i in range(len(test_Y_index)):
    test_Y_index_t[i] = corr_dict[test_Y_index[i]]
print(testvals,tvals)
# print(train_Y_index,train_Y_index_t)


# In[115]:


print(alldata[1][0][0])
# for i in range(0,11):
#     print(np.unique(np.array(alldata[i][1])))
# for i in range(1,11):
#     print(str(i),end=' ')
#     for val in range(len(alldata[i][1])):
#         alldata[i][1][val] = alldata[i][1][val] - (11*i)
pickle.dump(alldata, open('allfeatures.p','wb'))


# In[3]:


print("Saving vectors")
pickle.dump(alldata[:][0],open('onlyvecs.p','wb'))
print("Saving class labels")
pickle.dump(alldata[:][1],open('onlyclasses.p','wb'))
print("Saving records")
pickle.dump(alldata[:][2],open('onlyrecords.p','wb'))


# In[2]:


alldata = pickle.load(open('allfeatures.p','rb'))


# In[3]:


import torch.nn as nn
from torch.utils.data import TensorDataset
from torch import optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
scaler = pickle.load(open('fitted_scaler.p','rb'))


# In[55]:



kk = None
for model_num in range(1,11):
    #~~~FFNN Torch Data Initialization~~~
#     input_size = 10659
    input_size=250
    num_classes = 11
#     train_X = np.array(alldata[model_num-1][0])
#     train_Y_index_t = np.array(alldata[model_num-1][1])-(11*(model_num-1))
    print("Creating Cross-Validation Sets")
    train_X_list = []
    train_Y_list = []
    for x in range(0,10):
        if model_num-1 != x:
#             train_X_list.append(np.array(alldata[x][0]))
            train_X_list.append(np.array(alldata[x][0])[:,-250:])
            train_Y_list.append(np.array(alldata[x][1]))
    train_X = np.concatenate(train_X_list,axis=0)
    train_Y_index_t = np.concatenate(train_Y_list,axis=0)
    del train_X_list
    del train_Y_list

#     print('Normalizing data')

#     train_X_t = scaler.transform(train_X)
    train_X_t = train_X
    train_X_t = torch.tensor(train_X_t)
    train_Y_t = torch.tensor(train_Y_index_t)
#     test_Y_t = torch.tensor(test_Y_index_t)

#     print(np.unique(train_Y_index_t))
    #move batch norm to before ReLU
    layers = [nn.Linear(input_size,200),
              nn.BatchNorm1d(200),
              nn.LeakyReLU(),
              nn.Dropout(0.3),
              nn.Linear(200,200),
              nn.LeakyReLU(),
              nn.Dropout(0.3),
              nn.Linear(200,num_classes)]

    model = nn.Sequential(*layers)

    #move model to gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    #create dataset and dataloader
    bs = 512 #batch size
    train_ds = TensorDataset(train_X_t, train_Y_t)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=bs,
                                shuffle=True)
#     print(np.unique(train_Y_index_t))
    kk=compute_class_weight('balanced',np.unique(train_Y_index_t),train_Y_index_t)
    train_weights=torch.tensor(kk).float()

    criterion = nn.CrossEntropyLoss(weight=train_weights)
    optimizer = optim.AdamW(model.parameters(),lr = 0.0001)

    #Problem: all outputs in a batch are the same - dataloader issue? maybe just try keras

    epochs = 10

    model.train()
    tr_loss = []
    print("Training Model ",model_num)
    for epoch in range(epochs):
        running_loss = 0.0
        print('Epoch ',epoch,end = '\r')
        for i, data in enumerate(trainloader, 0):

            inputs,labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #print("Loss: ",running_loss)

    print('\n')
    s = "embed_models/1_model_"+str(model_num)
    save_pth = os.path.abspath(s)
    torch.save(model,save_pth)
    


# In[60]:


def predict_ensemble(model_version,test_X,embed_only = False):
    indiv_pred = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for model_num in range(1,11):
        if embed_only:
            path = "np_models/" + str(model_version) + "_model_" + str(model_num)
        else:
            path = "np_models/" + str(model_version) + "_model_" + str(model_num)
        abs_path = os.path.abspath(path)
        test_model = torch.load(abs_path).to(device)
        test_model.eval()
        y_pred = test_model(test_X_t.to(device).float())
        s = nn.Softmax(dim = 1)
        soft = s(y_pred).detach().numpy()
        indiv_pred.append(soft)
    total_pred = np.sum(indiv_pred, axis=0)
    return total_pred,indiv_pred
    
        


# In[61]:


test_X = np.array(alldata[10][0])
test_X_t = scaler.transform(test_X)
test_X_t = torch.tensor(test_X_t)

# total_pred,indiv_pred=predict_ensemble(1,test_X_t)
embed_pred, _ = predict_ensemble(1,test_X_t[:,-250:],embed_only=True)


# In[22]:


total_pred_gt8 = np.where(np.amax(total_pred,axis=1)>=8,1, 0)
print(total_pred_gt8.shape)


# In[62]:


from sklearn.metrics import classification_report
test_Y = np.array(alldata[10][1])
correct = ['major_capsid','minor_capsid','baseplate','major_tail','minor_tail',
         'portal','tail_fiber','shaft','collar','HTJ','others']

# print('Classification Report: PhANNs 2.0 Ensemble\n')
# print(classification_report(test_Y,
#                             np.argmax(total_pred,axis=1),target_names=correct))

# # where prediction score > 8
# print('Classification Report: Ensemble Score >= 8\n')
# print(classification_report(test_Y[total_pred_gt8 == 1],
#                             np.argmax(total_pred[total_pred_gt8==1],axis=1),target_names=correct))

print('Classification Report: Embedding-Only Ensemble\n')
print(classification_report(test_Y,
                            np.argmax(embed_pred,axis=1),target_names=correct))


# In[64]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# cm = confusion_matrix(np.around(test_Y_t.detach().numpy()),
#                                np.argmax(y_pred.detach().numpy(),axis=1),normalize='true')
# cm = confusion_matrix(np.around(test_Y[total_pred_gt8 == 1]),
#                                np.argmax(total_pred[total_pred_gt8 == 1],axis=1),normalize='true')
cm = confusion_matrix(np.around(test_Y),
                               np.argmax(embed_pred,axis=1),normalize='true')

cmtable = sns.heatmap(cm,annot=True,fmt=".2f",cmap='coolwarm',xticklabels=correct,yticklabels=correct)
# cmtable = sns.heatmap(cm,annot=True,cmap='coolwarm',xticklabels=correct,yticklabels=correct)
sns.set(rc={'figure.figsize':(16,10)})
sns.set(font_scale=1.3)
# plt.xlabel('Predicted class')
# plt.ylabel('True class')
cmtable.set_xlabel('Predicted class')
cmtable.set_ylabel('True class')
cmtable.set_title('Embedding-Only Ensemble Confusion Matrix')
# plt.title('PhANNs + Embeddings Confusion Matrix')
plt.show()

cmtable.figure.savefig('embedding_ensemble_cm.png')


# In[8]:


train_X_all_l = []
train_Y_all_l = []
for x in range(0,10):
        train_X_all_l.append(np.array(alldata[x][0]))
        train_Y_all_l.append(np.array(alldata[x][1]))
train_X_all = np.concatenate(train_X_all_l,axis=0)
train_Y_all = np.concatenate(train_Y_all_l,axis=0)
print(train_X_all.shape)
train_X_all = train_X_all[:,-250:]


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
print(train_X_all.shape)
knn_classifier = KNeighborsClassifier(weights='distance',n_jobs=16)
knn_classifier.fit(train_X_all,train_Y_all)


# In[ ]:


y_pred_knn = knn_classifier.predict(test_X[:,-250:])


# In[ ]:


# pickle.dump(y_pred_knn,open('knn_predictions.p','wb'))


# In[27]:


y_pred_combine = np.where(np.argmax(total_pred,axis=1) == y_pred_knn,1,0)


# In[41]:


print('Classification Report: KNN Classifier w/Embeddings\n')
print(classification_report(test_Y,
                            y_pred_knn,target_names=correct))

print('Classification Report: PhANNs 2.0 + KNN Classifier w/Embeddings\n')
print(classification_report(test_Y[y_pred_combine==1],
                            np.argmax(total_pred[y_pred_combine==1],axis=1),target_names=correct))

print('Classification Report: PhANNs 2.0 Score >= 8 + KNN Classifier\n')
print(classification_report(test_Y[((y_pred_combine==1) & (total_pred_gt8 == 1))],
                            np.argmax(total_pred[((y_pred_combine==1) & (total_pred_gt8 == 1))],axis=1),
                            target_names=correct))


# In[50]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(np.around(test_Y),
                               y_pred_knn,normalize='true')

# #Row norm ensemble + KNN
# cm = confusion_matrix(np.around(test_Y[y_pred_combine >= 0]),
#                                y_pred_combine[y_pred_combine >=0],normalize='true')

# #Non norm ensemble + KNN
# cm = confusion_matrix(np.around(test_Y[y_pred_combine >= 0]),
#                                y_pred_combine[y_pred_combine >=0],normalize=None)

#ensemble >=8 + KNN
# cm = confusion_matrix(np.around(test_Y[((y_pred_combine==1) & (total_pred_gt8 == 1))]),
#                                np.argmax(total_pred[((y_pred_combine==1) & (total_pred_gt8 == 1))],axis=1),
#                                          normalize=None)

cmtable = sns.heatmap(cm,annot=True,fmt=".2f",cmap='coolwarm',xticklabels=correct,yticklabels=correct)
# cmtable = sns.heatmap(cm,annot=True,cmap='coolwarm',xticklabels=correct,yticklabels=correct)
sns.set(rc={'figure.figsize':(16,10)})
sns.set(font_scale=1.3)

cmtable.set_xlabel('Predicted class')
cmtable.set_ylabel('True class')
cmtable.set_title('KNN Confusion Matrix')
# plt.title('PhANNs + Embeddings Confusion Matrix')
plt.show()

# cmtable.figure.savefig('phanns_2_ensemble_knn.png')
# cmtable.figure.savefig('phanns_2_ensemble_knn_unnorm.png')
cmtable.figure.savefig('knn_norm_cm.png')


# In[184]:




epochs = 10

model.train()
tr_loss = []
for epoch in range(epochs):
    running_loss = 0.0
    print('Epoch ',epoch)
    for i, data in enumerate(trainloader, 0):
        
        inputs,labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("\t","Loss: ",running_loss)
    
print('Finished Training')


# In[117]:


#~~~Testing~~~
model.eval()
test_ds = TensorDataset(test_X_t, test_Y_t)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=1,
                            shuffle=True)

correct = 0
total = 0

with torch.no_grad():
    testing = True
    for data in testloader:
        vecs, labels = data[0].to(device), data[1].to(device)
        outputs = model(vecs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if testing:
            print(outputs)
            testing = False
        
print('Accuracy = ',100 * correct / total,'%')


# In[185]:


model.eval()
y_pred = model(test_X_t.to(device).float())


# In[198]:


from sklearn.metrics import classification_report
correct = ['major_capsid','minor_capsid','baseplate','major_tail','minor_tail',
         'portal','tail_fiber','shaft','collar','HTJ','others']
print(classification_report(test_Y_t.detach().numpy(),
                            np.argmax(y_pred.detach().numpy(),axis=1),target_names=correct))


# In[210]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# cm = confusion_matrix(np.around(test_Y_t.detach().numpy()),
#                                np.argmax(y_pred.detach().numpy(),axis=1),normalize='true')
cm = confusion_matrix(np.around(test_Y_t.detach().numpy()),
                               np.argmax(y_pred.detach().numpy(),axis=1))

cmtable = sns.heatmap(cm,annot=True,fmt=".0f",cmap='coolwarm',xticklabels=correct,yticklabels=correct)
# cmtable = sns.heatmap(cm,annot=True,cmap='coolwarm',xticklabels=correct,yticklabels=correct)
sns.set(rc={'figure.figsize':(16,10)})
sns.set(font_scale=1.3)
# plt.xlabel('Predicted class')
# plt.ylabel('True class')
cmtable.set_xlabel('Predicted class')
cmtable.set_ylabel('True class')
cmtable.set_title('PhANNs + Embeddings Confusion Matrix')
# plt.title('PhANNs + Embeddings Confusion Matrix')
plt.show()

cmtable.figure.savefig('phanns_with_embeddings_cm_not_normal.png')


# In[196]:


# Save the model
PATH = './phann_model.pth'
torch.save(model, PATH)


# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model


# In[103]:


# test_Y_index = test_Y
# train_Y_index = train_Y


# In[101]:


# train_X = np.array(train_X)
# train_Y = np.array(train_Y)
# test_X=np.array(test_X)
# test_Y=np.array(test_Y)


# In[ ]:





# In[10]:


f_num=train_X.shape[1]
num_of_class=max(train_Y_index)+1
print(f_num,num_of_class)


# In[13]:


train_Y = np.eye(num_of_class)[train_Y_index]
test_Y  = np.eye(num_of_class)[test_Y_index]
print(test_X.shape)
print(test_Y.shape)
es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=5, min_delta=0.02 )
mc = ModelCheckpoint('phannmodel1.h5',monitor='val_loss', mode='min', save_best_only=True, verbose=1)
model = None


# In[15]:


kk=compute_class_weight('balanced',range(num_of_class),train_Y_index)
train_weights=dict(zip(range(num_of_class),kk))
model = Sequential()
opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
model.add(Dense(f_num, input_dim=f_num, kernel_initializer='random_uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_of_class,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(train_weights)
# history = model.fit(train_X, train_Y, validation_split=.2 , epochs=120, 
#           batch_size=256, verbose=1,class_weight=train_weights,callbacks=[es,mc])
history = model.fit(train_X, train_Y, validation_split=.2 , epochs=120, 
          batch_size=5000, verbose=1,callbacks=[es,mc])


# In[16]:


yhats = model.predict(test_X,verbose=2)


# In[19]:


yhats_v=np.array(yhats)
# predicted_Y=np.sum(yhats_v, axis=0)
predicted_Y_index = np.argmax(yhats_v,axis=1)
print(yhats_v)


# In[20]:


# labels_names = files
print(classification_report(test_Y, predicted_Y_index))


# In[ ]:




