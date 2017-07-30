import gensim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import configparser as cp

config = cp.RawConfigParser()
config.read('docvec1.cfg')

train_file = config.get('SectionOne', 'Train_file')
test_file = config.get('SectionOne', 'Test_file')

def read_corpus(fname,token_only = False):
	with open(fname, encoding="utf-8") as f:
		for i, line in enumerate(f):
			tmp = line.strip().split("\t")
			if len(tmp)>1:
				yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(tmp[1]), [i])
			else:
				yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(tmp[0]))

train_corpus = list(read_corpus(train_file))
test_corpus = list(read_corpus(test_file))
#print(train_corpus[1])
#print(test_corpus[1])
#model = gensim.models.doc2vec.Doc2Vec(train_corpus,size=200, window = 6, min_count=3, iter=50,workers=2)
#model.build_vocab(train_corpus)


def hash(astring):
   return ord(astring[0])

######Training Doc Vec Model######
"""
model = gensim.models.Doc2Vec(size=200, window=4,sample = 1e-4,negative=10, min_count=3, workers=2,hashfxn=hash) # use fixed learning rate

model.build_vocab(train_corpus)
for epoch in range(5):
    model.train(train_corpus,total_examples=len(train_corpus),epochs=100)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    print("Epoch %s completed...." %(epoch))

model.save(config.get('SectionOne', 'Model_Path'))
"""

def read_tags_sent(fname):
	with open(fname, encoding="utf-8") as f:
		label=[]
		sentence=[]
		for line in f:
			tmp = line.strip().split("\t")
			if len(tmp)> 1:
				label.append(tmp[0])
				sentence.append(tmp[1])
			else:
				sentence.append(tmp[0])
	return label,sentence


def create_vectors(corpus):
	model_load = gensim.models.doc2vec.Doc2Vec.load(config.get('SectionOne', 'Model_Path'))
	corpus_features=[]
	for i in range(0,len(corpus)):
		corpus_features.append(model_load.infer_vector(corpus[i][0]))
	return corpus_features

train_data_features = create_vectors(train_corpus)
train_target,train_sentence = read_tags_sent(train_file)
#print(train_target[1])
test_data_features = create_vectors(test_corpus)
test_target,test_sentence = read_tags_sent(test_file)

#forest = RandomForestClassifier(n_estimators = 100) 
lr = LogisticRegression()
#svc1 = SVC(kernel='rbf')

second_model = lr.fit(train_data_features,train_target)
test_prediction = lr.predict(test_data_features)

if len(test_target)!=0:
	print("Accuracy test- ")
	print(accuracy_score(test_target, test_prediction))
	print("F1 Score test- ")
	print(f1_score(test_target, test_prediction, average='weighted'))
	main_data = pd.DataFrame({'Orig_Val': test_target,'DocVecPred': test_prediction,'Sentence': test_sentence})
	main_data.to_csv(config.get('SectionOne', 'Save_Result'), index=False, encoding="utf-8",header = True,sep="\t")
else:
	main_data = pd.DataFrame({'DocVecPred': test_prediction,'Sentence': test_sentence})
	main_data.to_csv(config.get('SectionOne', 'Save_Result'), index=False, encoding="utf-8",header = True,sep="\t")
