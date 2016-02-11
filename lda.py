import re, nltk        
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora, models
import gensim

fo = open('test.txt')
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def tokenize(text):
	text = text.lower()
	text = re.sub("[^a-zA-Z]", " ", text) 
	# text = re.sub("[http://]", " ", text)
	text = re.sub(" +"," ", text) 
	text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ",text) 
	text = re.sub("\\b[a-zA-Z0-9]{0,1}\\b"," ",text) 
	tokens = nltk.word_tokenize(text.strip())
	# Uncomment next line to use stemmer
	# tokens = stem_tokens(tokens, stemmer)
	tokens = nltk.pos_tag(tokens)
	return tokens

stopset = stopwords.words('english')
# freq_words = ['comment']
# for i in freq_words :
#     stopset.append(i)

text_corpus = []
for doc in fo :
	temp_doc = tokenize(doc.strip())
	current_doc = []
	for word in range(len(temp_doc)) :

		if (temp_doc[word][0] not in stopset) and (temp_doc[word][1] == 'NN' or temp_doc[word][1] == 'NNS' or temp_doc[word][1] == 'NNP' or temp_doc[word][1] == 'NNPS'):
			current_doc.append(temp_doc[word][0])

	text_corpus.append(current_doc)


dictionary = corpora.Dictionary(text_corpus)
# print dictionary
#dictionary.save('myDict.dict')
# print dictionary.token2id
corpus = [dictionary.doc2bow(text) for text in text_corpus]
# print corpus


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1, id2word = dictionary,passes=100)
for topics in ldamodel.print_topics(num_topics=1, num_words=10) :
	print topics,"\n"
