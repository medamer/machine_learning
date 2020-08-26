import logging
import random
import pandas
import sqlite3
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field, validator
import pickle

log = logging.getLogger(__name__)
router = APIRouter()
​
"""
Code below is in progress, currently commented out to prevent errors
when deploying to Heroku
"""
​# Import the pickled model:
model = pickle.load(open("model.pkl", "rb"))
​
class Input(BaseModel):
    """Use this data model to parse the request body JSON."""
    input_phrase: str
​
​
@router.post('/predict')
async def test_prediction(user_input: Input):
    conn = sqlite3.connect('cannabis.sqlite3')
    curs = conn.cursor()
    pred = 687  # Stable prediction before the model goes into place
    #pred = model.predict(user_input.input_phrase)
    pred = model.transform(user_input.input_phrase)
    # find similar effects:
    pred = pred.todense()
    pred = nn.kneighbors(pred, return_distance=False)
    pred = pred[0][0]
    #
    query_strain = curs.execute(f"SELECT * FROM Cannabis WHERE Strain_ID == {pred} ORDER BY Rating")
    strain = curs.fetchall()
    keys = ['ID', 'Strain_id', 'Name', 'Type', 'Rating', 'Effects', 'Description', 'Flavors', 'Neighbors']
    suggestion = {k: v for k, v in zip(keys, strain[0])}
    for key in ['Effects', 'Flavors', 'Neighbors']:
        suggestion[key] = suggestion[key].split(',')
​
    return JSONResponse(content=suggestion)
​
​
@router.get('/init_db')
async def init_db():
    df = pd.read_csv('cannabis_new.csv')
    df = df.rename(columns={'Index': 'Strain_ID'})
    conn = sqlite3.connect('cannabis.sqlite3')
    curs = conn.cursor()
    curs.execute("DROP TABLE IF EXISTS Cannabis")
    df.to_sql('Cannabis', con=conn)