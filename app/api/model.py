import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


nlp = spacy.load('en_core_web_lg')

# Load the dataset:
df = pd.read_csv('cannabis_new.csv')


# Define a function to tokenize the text:
def tokenizer(text):
    doc=nlp(text)
    return [token.lemma_ for token in doc if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON')]

# Build the model:
model = TfidfVectorizer(stop_words = 'english',
                        ngram_range = (1,2),
                        max_df = .95,
                        min_df = 3,
                        tokenizer = tokenizer)

# Fit and transform the data:
dtm = model.fit_transform(df['Effects'])

# Get features:
dtm = pd.DataFrame(dtm.todense(), columns = model.get_feature_names())

model_t = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
model_t.fit(dtm)



pickle.dump(model, open("../model.pkl", "wb"))
pickle.dump(model_t, open("../model_t.pkl", "wb"))
