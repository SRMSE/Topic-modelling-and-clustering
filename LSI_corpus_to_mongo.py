import re, nltk        
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import gensim
import pymongo

client = pymongo.MongoClient()
db = client['lsi']

def lsi_main(link,article):
	
	i=i=1
	# print "inside"
	article = str(filter(lambda x:ord(x)>31 and ord(x)<128,article))
	fo= article.strip()
	
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
	
	temp_doc = tokenize(fo.strip())
	current_doc = []
	for word in range(len(temp_doc)) :

		if (temp_doc[word][0] not in stopset) and (temp_doc[word][1] == 'NN' or temp_doc[word][1] == 'NNS' or temp_doc[word][1] == 'NNP' or temp_doc[word][1] == 'NNPS'):
			current_doc.append(temp_doc[word][0])

	text_corpus.append(current_doc)

	# print text_corpus

	dictionary = corpora.Dictionary(text_corpus)
	corpus = [dictionary.doc2bow(text) for text in text_corpus]
	
	# tfidf = models.TfidfModel(corpus)
	# corpus = tfidf[corpus]

	lsi = models.LsiModel(corpus,id2word=dictionary, num_topics=1)

	
	topics = lsi.print_topics(1)[0][1]
	print "link :: ",link
	
	db.topics.insert({"topics":topics,"doc":article,"link":link})
	'''
	# for inverse index
	topic_list = topics.split("+")
	
	for ele in topic_list:
		prob = float(str(ele.split("*")[0]))
		topic = str(ele.split("*")[1])
		print "prob ",prob, " topic",topic,"\n"
		db.topics.update({"topic":topic},{"$set":{str(prob):fo}},True)
		
		# print "prob ",(float(str(prob)))," topic ",str(topic) 
	'''
	# query = ["the bank of port".lower().split(' ')]

	# vec_query = [dictionary.doc2bow(text) for text in query] 
	# index = similarities.MatrixSimilarity(lsi[corpus])
	# similarities = index[lsi[vec_query]]
	# print similarities
