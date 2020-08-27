import logging
import random
import pandas
import sqlite3
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field, validator
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib
import spacy
from sklearn.neighbors import NearestNeighbors

nlp = spacy.load('en_core_web_lg')
log = logging.getLogger(__name__)
router = APIRouter()

"""
Code below is in progress, currently commented out to prevent errors
when deploying to Heroku
"""

####################################################################################################
# Define a function to tokenize the text:
def tokenizer(text):
    doc=nlp(text)
    return [token.lemma_ for token in doc if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON')]

model = TfidfVectorizer(stop_words = 'english',
                        ngram_range = (1,2),
                        use_idf=True,
                        smooth_idf=True,
                        #max_df = .5,
                        #min_df = .1,
                        tokenizer = tokenizer)

df = pd.read_csv('./app/api/cannabis_new.csv')
dtm = model.fit_transform(df['Effects'])
dtm = pd.DataFrame(dtm.todense(), columns = model.get_feature_names())
model_t = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
model_t.fit(dtm)
#####################################################################################################



class Input(BaseModel):
    """Use this data model to parse the request body JSON."""
    input_phrase: str

@router.post('/predict')
async def test_prediction(user_input: Input):
    conn = sqlite3.connect('cannabis.sqlite3')
    curs = conn.cursor()
    user_input_ = [user_input.input_phrase]
    pred = model.transform(user_input_)
    pred = pred.todense()
    pred = model_t.kneighbors(pred, return_distance=False)
    pred = pred[0][0]
    query_strain = curs.execute(f"SELECT * FROM Cannabis WHERE Strain_ID == {pred} ORDER BY Rating")
    strain = curs.fetchall()
    keys = ['ID', 'Strain_id', 'Name', 'Type', 'Rating', 'Effects', 'Description', 'Flavors', 'Neighbors']
    suggestion = {k: v for k, v in zip(keys, strain[0])}
    for key in ['Effects', 'Flavors', 'Neighbors']:
        suggestion[key] = suggestion[key].split(',')

    return JSONResponse(content=suggestion)

@router.get('/init_db')
async def init_db():
    df = pd.read_csv('cannabis_new.csv')
    df = df.rename(columns={'Index': 'Strain_ID'})
    conn = sqlite3.connect('cannabis.sqlite3')
    curs = conn.cursor()
    curs.execute("DROP TABLE IF EXISTS Cannabis")
    df.to_sql('Cannabis', con=conn)