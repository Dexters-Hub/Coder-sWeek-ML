import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.linear_model import SGDClassifier
import logging
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings;warnings.filterwarnings('ignore')


df_corpus = pd.read_csv("labeled_data.csv")
df_corpus.shape

df_corpus.head()

df_corpus['class'].value_counts()

df_corpus.drop_duplicates(inplace=True)
df_corpus.shape

df_corpus['class'] = df_corpus['class'].apply(lambda x: 1 if (x==0  or x==1) else 0)


sns.barplot(['Non Toxic', 'Toxic'], df_corpus['class'].map({0:"Non Toxic", 1: "Toxic"}).value_counts(ascending=True), alpha=0.8,palette="vlag")

plt.title('Count of Toxic Comment of Dataset 1')


df_corpus2 = pd.read_csv("train.csv", error_bad_lines=False)
df_corpus2.shape

df_corpus2.head()

df_corpus2.columns = ["id", "class", "tweet"]


sns.barplot(['Non Toxic', 'Toxic'], df_corpus2['class'].map({0:"Non Toxic", 1: "Toxic"}).value_counts(), alpha=0.8,palette="vlag")


plt.title('Count of Toxic Comment of Dataset 2')


df_corpus_final = pd.concat([df_corpus[['class', 'tweet']], df_corpus2[['class', 'tweet']]])

df_corpus_final.head()

df_corpus_final.reset_index( drop=True, inplace=True)

df_corpus_final['class'].value_counts().plot(kind='bar')


sns.barplot(['Non Toxic', 'Toxic'], df_corpus_final['class'].map({0:"Non Toxic", 1: "Toxic"}).value_counts(), alpha=0.8,palette="vlag")
plt.title('Count of Toxic Comment All Datasets ')


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def decontracted(phrase):
    

    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    
    
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

my_sw = ['rt', 'ht', 'fb', 'amp', 'gt']
def black_txt(token):
  if token == 'u':
    token = 'you'
  return  token not in stop_words_ and token not in list(string.punctuation) and token not in my_sw

def cleaner(word):
  

  word = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 
                '', word, flags=re.MULTILINE)
  

  word = decontracted(word)
  

  word = re.sub(r'(@[^\s]*)', "", word)
  word = re.sub('[\W]', ' ', word)
  

  list_word_clean = []
  for w1 in word.split(" "):
    if  black_txt(w1.lower()):
      word_lemma =  wn.lemmatize(w1,  pos="v")
      list_word_clean.append(word_lemma)

  

  word = " ".join(list_word_clean)
  word = re.sub('[^a-zA-Z]', ' ', word)
  return word.lower().strip()

df_corpus.iloc[24579]['tweet']

cleaner(df_corpus.iloc[24579]['tweet'])



for idx in df_corpus.tail(15).index:
  print(cleaner(df_corpus.iloc[idx]['tweet']),'\n'  , df_corpus.iloc[idx]['tweet'], idx)
  print("************")



from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model

X = df_corpus_final['tweet']
y = df_corpus_final['class']
encoder = LabelEncoder()
y = encoder.fit_transform(y)
Y = np_utils.to_categorical(y)

from sklearn.base import BaseEstimator, TransformerMixin

import spacy

nlp = spacy.load('en_core_web_lg')


tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(df_corpus_final['tweet'])
embeddings_index = np.zeros((30000 + 1, 300))
for word, idx in tokenizer.word_index.items():
    try:
          embedding = nlp.vocab[word].vector
          embeddings_index[idx] = embedding
    except:
      pass


embeddings_index[1]


class KerasTextClassifier(BaseEstimator, TransformerMixin):
  
  def __init__(self, max_words=30000, input_length=20, emb_dim=300, n_classes=2, epochs=15, batch_size=64, emb_idx=0):
    self.max_words = max_words
    self.input_length = input_length
    self.emb_dim = emb_dim
    self.n_classes = n_classes
    self.epochs = epochs
    self.bs = batch_size
    self.embeddings_index = emb_idx
    self.tokenizer = Tokenizer(num_words=self.max_words+1, lower=True, split=' ')
    self.model = self._get_model()
    return self.model.summary()
    
  def _get_model(self):
    input_text = Input((self.input_length,))
    text_embedding = Embedding(input_dim=self.max_words+1, output_dim=self.emb_dim, input_length=self.input_length, 
                               mask_zero=False, weights=[self.embeddings_index], trainable=False)(input_text)
    text_embedding = SpatialDropout1D(0.4)(text_embedding)
    bilstm =(LSTM(units=50,  recurrent_dropout=0.2, return_sequences = True))(text_embedding)
    x = Dropout(0.2)(bilstm)
    x =(LSTM(units=50,  recurrent_dropout=0.2, return_sequences = True))(x)
    x = Dropout(0.2)(x)
    x =(LSTM(units=50,  recurrent_dropout=0.2))(x)
    out = Dense(units=self.n_classes, activation="softmax")(x)
    model = Model(inputs=[input_text],outputs=[out])
    model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])
    return model
  
  def _get_sequences(self, texts):
    seqs = self.tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=self.input_length, value=0)
  
  def _preprocess(self, texts):
    return [cleaner(x) for x in texts]
  
  def fit(self, X, y):
    
    self.tokenizer.fit_on_texts(X)
    self.tokenizer.word_index = {e: i for e,i in self.tokenizer.word_index.items() if i <= self.max_words}
    self.tokenizer.word_index[self.tokenizer.oov_token] = self.max_words + 1
    seqs = self._get_sequences(self._preprocess(X))
    self.model.fit([seqs ], y, batch_size=self.bs, epochs=self.epochs, validation_split=0.1)
  
  def predict_proba(self, X, y=None):
    seqs = self._get_sequences(self._preprocess(X))
    return self.model.predict(seqs)
  
  def predict(self, X, y=None):
    return np.argmax(self.predict_proba(X), axis=1)
  
  def score(self, X, y):
    y_pred = self.predict(X)
    return accuracy_score(np.argmax(y, axis=1), y_pred)

text_model = KerasTextClassifier(emb_idx= embeddings_index)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.1, random_state = 40, stratify=Y)



text_model.fit(x_train, y_train)



text_model.score(x_test, y_test)



from IPython.display import display, HTML





import eli5
from eli5.lime import TextExplainer

for idx in x_test.index[190:210]:
  te = TextExplainer(random_state=42)
  te.fit(cleaner(x_test[idx]), text_model.predict_proba, )
  print("Real Class:",  ["Non Toxic" if x == 0 else "Toxic" for x in [df_corpus_final.iloc[idx]['class']]])
  print("Text uncleaned tweet:", df_corpus_final.iloc[idx]['tweet'])
  print("ELI5 Predicted Class:")
  HTML(display((te.show_prediction(target_names=[ 'Non Toxic','Toxic',]))))
  
  import pickle
  pickle.dump(text_model, open('toxic.pickle', 'wb'))