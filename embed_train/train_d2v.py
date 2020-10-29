import numpy as np
import pickle
import gensim
from Bio.Seq import Seq
from Bio import SeqIO 
import os
import itertools

files = ["phage_euk_dr100",
         "phage_prok_dr100",
         "phage_arch_dr100",
         "vir_euk_dr100",
         "vir_prok_dr100",
         "vir_arch_dr100"]


training_data = []
kmer_size = 3
for file in files:
    print("Converting " + file + ".fasta")
    f_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"fasta",file + ".fasta")
    records = list(SeqIO.parse(f_path,"fasta"))
    for rec in records:
        #format the sequence string by making uppercase...
        #replace X with A, J with L, * with A, Z with E, and B with D
        string = str(rec.seq).upper()
        string = string.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')

        #Different non-overlapping windows
        for i in [0,1,2]:
            kmer_strings = [string[k:k+kmer_size] for k in range(i,len(string),kmer_size)]
            training_data.append(gensim.models.doc2vec.TaggedDocument(kmer_strings,[str(rec.id)+str(i)]))

alphabet=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
#all_kmers = [gensim.models.doc2vec.TaggedDocument(list(itertools.product(alphabet,repeat=kmer_size)),[0])]
all_kmers=[]
for km in itertools.product(alphabet,repeat=kmer_size):
    all_kmers.append("".join(km))
vocab_list = [gensim.models.doc2vec.TaggedDocument(all_kmers,[0])]

d2v_model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=250, epochs=10, max_vocab_size=20**kmer_size, hs=0,negative=20)

d2v_model.build_vocab(training_data)

d2v_model.train(training_data, total_examples=len(training_data),epochs=d2v_model.epochs)

# +
import tempfile

with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    d2v_model.save(temporary_filepath)
# -
type(d2v_model)


from gensim.test.utils import get_tmpfile
fname = get_tmpfile("d2v_model1")
d2v_model.save(fname)


import pickle
output_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "d2v_model1.p")
pickle.dump(d2v_model,open(output_path,"wb"),protocol=4)

test = pickle.load(open(output_path,'rb'))
print(type(test))
test = 0

sequence_test = "MAENKTNIRISGDVSDLNASVESAKRSLAGLGDAAAQTGKKAQQSGNQATDSYRKVSNSAKKAGDDADAASKKMERSANSLRASIERTIAVQEAGGRNTSNFFRSLLTQRGLDSGQYAHQFEPLLKRLDELNNAAKKTAETINKTSDATENAAKRYEDIIRRTIAAQEAGGRSTSGYFKSLIEQEGLDPKRFDGLFNRLDEVNAASKKASDETEVNARKIEASIQRIVAAETAGGRSNRKYFETLAQQSGLDTKRLEPLLKQLDEVNSNSRKIAEAREADSKKIEASIQRIINAETAGTLSNRQYFESLINQRGLDPKRFDPLLQKLDALDKRTKGLTISYGQYQNALRMMPAQFTDIVTQLAGGQSPFLIAIQQGGQIRDSFGGFGNMFKGLLSMITPARLAIGGLVGVIGAVGAAFIQGSKESDFFRKAVILAGGSSSVTAGQIQMMAAKIGDSTGAISEAREALTSLISTGAAVSETFEQVATAIAYNSEMTGQKVEELVKQFAKVKEEPVKAVVELSQNYDTLTVAVYEQAKALTEAGRKGDAVILVQNKLAQGVVEAGRRTWESAGLMEKGWLTVKKAAEMAWDAMKGIGREDPLEKQLQSVLQQIAKLEAQKKGDSFFGTAYDDDIAKLKKEAVEIQRKIKSDSDAQKARQQQAEAVAKRAEMDKQADSVIKANQTPVERIDEQIKQARELEKYYRSIKDDKIAANKADQIALDIGRMQKDRAEAVKKANEKGKPRKHDQRLMNDTVRLQASRYNYSGLERQYGLPAGLLAAISMQESGGNPKALSIAGARGLFQFMPGTAKRFGVNVHDPASSADGAAKYLSYLLKFFKGDLIKAISAYNAGEGTISNIGKPTKGGRIRQLPTETRKYTPMVLKRMAAYNNQSDDGSADYASDYLAKLQELAKKRLEIEKSFYTKREKLAADYQERLDKINEAGFDDETKQKYLNDAKEAYERDLEAYDDAIKRKLQSAWDFNKNAIELIHERTEAERKEIERNFELTKEQQAELIKALHAREAAEIDDVLGLSNVQAAVKQIQILTQALKENRISEARFKSKIGKIDIIRDYKDAMNWGKQDDIFESLADQYTDNIVKIEDYYRLQAELAKDNAADLVKIERQKLEELDAMQKTWMQQNLSAYLNYNEQIFGSMSSMLEESVGKHSTAYRLMLATQKAFAIASSVIAIQNAIAQASAAPFPSNLAAMATVAAETANIVSSIAAVSAGFSSGGYTGDGGKYEPAGIVHRGEFVLNQADVRNMGGVAGIERLRALAGGSSRGYADGGAVGRSVIGNMTANPAIAMGGVHQTITVNGNPDNATMQAIENAAKRGAQMGYQQVARHLATGQGDVSKALKGGWTTNRKLS"
sequence_test2 = "MANVIKTVLTYQLDGSNRDFNIPFEYLARKFVVVTLIGVDRKVLTINTDYRFATRTTISLTKAWGPADGYTTIELRRVTSTTDRLVDFTDGSILRAYDLNVAQIQTMHVAEEARDLTTDTIGVNNDGHLDARGRRIVNLANAVDDRDAVPFGQLKTMNQNSWQARNEALQFRNEAETFRNQAEGFKNESSTNATNTKQWRDETKGFRDEAKRFKNTAGQYATSAGNSASAAHQSEVNAENSATASANSAHLAEQQADRAEREADKLENYNGLAGAIDKVDGTNVYWKGNIHANGRLYMTTNGFDCGQYQQFFGGVTNRYSVMEWGDENGWLMYVQRREWTTAIGGNIQLVVNGQIITQGGAMTGQLKLQNGHVLQLESASDKAHYILSKDGNRNNWYIGRGSDNNNDCTFHSYVHGTTLTLKQDYAVVNKHFHVGQAVVATDGNIQGTKWGGKWLDAYLRDSFVAKSKAWTQVWSGSAGGGVSVTVSQDLRFRNIWIKCANNSWNFFRTGPDGIYFIASDGGWLRFQIHSNGLGFKNIADSRSVPNAIMVENE"
seqtest=sequence_test.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')
strlist = [seqtest[k:k+kmer_size] for k in range(0,len(seqtest),kmer_size)]
seqtest=sequence_test2.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')
strlist2 = [seqtest[k:k+kmer_size] for k in range(0,len(seqtest),kmer_size)]

inferred_vector1 = d2v_model.infer_vector(strlist)
sims = d2v_model.docvecs.most_similar([inferred_vector1])

inferred_vector2 = d2v_model.infer_vector(strlist2)

id_lookup = {}
for file in files:
    print(file)
    f_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"fasta",file + ".fasta")
    records = list(SeqIO.parse(f_path,"fasta"))
    for rec in records:
        id_lookup[str(rec.id)] = str(rec.description) + " " + file

for s in sims:
    seqid = s[0]
    seqid = seqid[0:len(seqid)-1]
    print(id_lookup[seqid],s[1])

d2v_model.docvecs.n_similarity(strlist,strlist2)


