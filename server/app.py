from flask import Flask, jsonify, request, abort, redirect
import sqlite3, uuid, json
from datetime import datetime
import math, urllib.parse
from flask.templating import render_template
from flask_cors import cross_origin
from celery import Celery
from query import Requester
from save_embedded_data import save_embedded_data

debug_flag = False
app = Flask(__name__)

db_path = r"./data/data.db"
MAIL_URL = r"https://uwmadison.co1.qualtrics.com/jfe/form/SV_6feG3xPXs0hRuYu"
SURVEY_URL = r"https://uwmadison.co1.qualtrics.com/jfe/form/SV_1NZcTVqNnDlq4cK"
SURVEY_URLS = set((
    r"https://uwmadison.co1.qualtrics.com/",
    r"https://pennstate.co1.qualtrics.com/"))

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['result_backend'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

app.config.update(
    CELERY_BROKER_URL='amqp://guest:guest@localhost:5672/',
    result_backend=None
)
celery = make_celery(app)

@celery.task(name="app.make_query")
def make_query(ip):
    requester = Requester()
    requester.request_store(ip)

@celery.task(name="app.save_qdata")
def save_qdata(data, ip, receive_utc):
    save_embedded_data(data, ip, receive_utc)

def initialize_dbs():
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        # FirstRequest: 1. check referrer; 2. store the first time (check wait time)
        cur.execute(
            '''CREATE TABLE if not exists UidRequest
                    (uid TEXT, social_source TEXT, IP TEXT, receive_utc INTEGER, http_header TEXT)
        ''')
        # IP column is what's recorded by qualtrics
        cur.execute(
            '''CREATE TABLE if not exists Qualtrics
                    (response_id TEXT, IP TEXT, receive_utc INTEGER, http_header TEXT)
        ''')

        conn.commit()


def is_valid_qualtrics_request(record: dict):
    return ('ip' in record) and ('response_id' in record)

def is_valid_qdata_request(record: dict):
    return (('responseId' in record) and ('field' in record)
        and ('value' in record) and ('qid' in record) )

def get_utcnow_second():
    return math.floor(datetime.now().timestamp())

@app.route("/gets/first-request", methods=["GET"])
@cross_origin(origins=[r".+\.qualtrics.com", r".+\.nemo-arxiv\.club", r".+\.psu\.edu"], methods="GET")
def respond_uuid():
    receive_utc = get_utcnow_second()
    source = request.args.get("Q_SocialSource")
    uid = uuid.uuid4().hex
    ip = request.headers.get("X-My-Origin-IP")
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO UidRequest (uid, IP, receive_utc, http_header, social_source)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (uid, ip, receive_utc, json.dumps(dict(request.headers)), source)
        )
        conn.commit()
    query = urllib.parse.urlencode({"Q_SocialSource": source, "uid": uid})
    url = SURVEY_URL + f"?{query}"
    response = {"url": url}
    return jsonify(response)


def url_remove_query(url: str):
    ind = url.rfind("?")
    if ind == -1:
        return url
    else:
        return url[:ind]


@app.route("/posts/qdata", methods=["POST"])
@cross_origin(origins=[r".+\.qualtrics.com", r".+\.nemo-arxiv\.club"], methods="POST")
def add_qdata_record():
    receive_utc = get_utcnow_second()
    data = request.json
    ip = request.headers.get("X-My-Origin-IP")
    if (not data) or (not is_valid_qdata_request(data)):
        abort(400)
    save_qdata.delay(data, ip, receive_utc)
    return jsonify({"status": "success"}), 201


@app.route("/posts/qualtrics", methods=["POST"])
@cross_origin(origins=[r".+\.qualtrics.com", r".+\.nemo-arxiv\.club"], methods="POST")
def add_qualtrics_record():
    receive_utc = get_utcnow_second()
    new_record = request.json
    ip = new_record.get("ip")
    if (not new_record) or (not is_valid_qualtrics_request(new_record)):
        abort(400)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO Qualtrics (response_id, IP, receive_utc, http_header)
            VALUES (?, ?, ?, ?)''',
            (new_record.get("response_id"), ip, receive_utc, json.dumps(dict(request.headers)))
        )
        conn.commit()
    make_query.delay(ip)
    return jsonify({"status": "success"}), 201


if __name__ == "__main__":
    initialize_dbs() 
    app.run(host="127.0.0.1", port=3111, debug=debug_flag)



