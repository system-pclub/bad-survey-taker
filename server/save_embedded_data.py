import os, json
from datetime import datetime
import math
from pprint import pprint
import sqlite3

db_path = r"./data/embedded_data.db"

def get_utcnow_second():
    return math.floor(datetime.now().timestamp())

def initialize_db(db_path=db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            '''CREATE TABLE if not exists EmbeddedData
                    (IP TEXT, receive_utc INTEGER, response_id TEXT, qid TEXT, field TEXT, value TEXT)
        ''')
        conn.commit()

def save_embedded_data(data, ip, receive_utc):
    initialize_db()
    qid = data.get("qid")
    response_id = data.get("responseId")
    field = data.get("field")
    value = data.get("value")
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO EmbeddedData (IP, receive_utc, response_id, qid, field, value)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (ip, receive_utc, response_id, qid, field, value)
        )
        conn.commit()

    

