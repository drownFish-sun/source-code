import random

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from transformers import BertModel,BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import torch
from sklearn import preprocessing



class emb:
    def __init__(self):
        self.prepare()

    def normalize(self, data_preprocessed):
        scaler = preprocessing.StandardScaler()
        data_preprocessed = scaler.fit_transform(data_preprocessed)
        return data_preprocessed
    def load_glove_embeddings(self, file_path):
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index
    def prepare(self):
        glove_file_path = 'glove/glove.6B.300d.txt'
        self.embeddings_index = self.load_glove_embeddings(glove_file_path)
        self.tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
        self.model = BertModel.from_pretrained("./bert-base-uncased")

    def TFIDF(self, sentences):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # feature_names = vectorizer.get_feature_names()
        tfidf_array = tfidf_matrix.toarray()
        return tfidf_array

    def w2v(self, sentences, mxlen):
        # tagged_documents = [TaggedDocument(sentence, [i]) for i, sentence in enumerate(sentences)]
        # model = Doc2Vec(tagged_documents, vector_size=mxlen, window=5, min_count=1, workers=4, epochs=40)
        # embeddings = []
        # for sentence in sentences:
        #     embeddings.append(model.infer_vector(sentence))
        # return np.array(embeddings)
        model = Word2Vec(hs=1, min_count=1, window=5, vector_size=100)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=5)
        embeddings = []
        for sentence in sentences:
            embedding = []
            for word in sentence:
                embedding.append(model.wv.get_vector(word).sum())
            if len(embedding) < mxlen:
                embedding.extend([0] * (mxlen - len(embedding)))
            embeddings.append(embedding)
        return np.array(embeddings)
    def GloVe(self, sentences):
        def get_word_vector(word, embeddings_index):
            vector = embeddings_index.get(word)
            if vector is not None:
                return vector
            else:
                return np.zeros(100)

        def log_to_vector(sentence, embeddings_index):
            word_vectors = [np.mean(get_word_vector(word.lower(), embeddings_index), axis=0) for word in sentence]
            # print(np.mean(word_vectors, axis=0).shape)
            return np.mean(word_vectors, axis=0)

        ret = np.zeros((len(sentences), 100))
        for i, sentence in enumerate(sentences):
            ret[i] = log_to_vector(sentence, self.embeddings_index)
        return ret
    def BERT(self, log_data):

        sentences = []
        for i in log_data:
            sentences.append(str(i))

        rets = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                rets.append(outputs['pooler_output'].cpu().detach().numpy()[0])
        # predictions = torch.argmax(logits, dim=-1)
        # print(outputs.shape)
        return rets

    def node_embedding(self, logs, num=4):
        sentences = []
        mxlen = 0
        for event in logs:
            i = event.split(' ')
            sentences.append(i)
            mxlen = max(mxlen, len(i))
        w2v_data = self.w2v(sentences, mxlen)
        tfidf_data = self.TFIDF(logs)
        # glove_data = self.GloVe(sentences)
        bert_data = self.BERT(logs)
        # return [normalize(np.array(w2v_data)), normalize(np.array(tfidf_data)), normalize(np.array(bert_data)), normalize(np.array(glove_data))]
        return [self.normalize(np.array(w2v_data)), self.normalize(np.array(bert_data)), self.normalize(np.array(tfidf_data))]
# def read(path):
#     data = pd.read_csv(path)
#
#     try:
#         label = data['label']
#     except KeyError:
#         label_ = data['Label']
#         label = np.zeros(len(label_), int)
#         for i in range(len(label_)):
#             label[i] = 0 if label_[i] == '-' else 1
#
#     EventTemplate = data['EventTemplate']
#     sentences = []
#     for event in EventTemplate:
#         i = event.split(' ')
#         sentences.append(i)
#     w2v_data = w2v(sentences, EventTemplate)
#     tfidf_data = TFIDF(sentences)
#     # glove_data = GloVe(sentences)
#     bert_data = BERT(EventTemplate)
#     lt = [i for i in range(0, data.shape[0])]
#     random.seed(10000)
#     random.shuffle(lt)
#     w2v_data = w2v_data[lt]
#     tfidf_data = tfidf_data[lt]
#     bert_data = bert_data[lt]
#     label = label[lt]
#
#     return w2v_data, tfidf_data, bert_data, label
    # print(glove_data.shape)
