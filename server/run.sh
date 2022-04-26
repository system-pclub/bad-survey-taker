# setsid ./venv/bin/python3 app.py >> app.log 2>&1 &
setsid /home/archiver/py_projects/flask_json/venv/bin/uwsgi -s 127.0.0.1:3111 --manage-script-name --mount /=app.app --master --threads 2 --stats 127.0.0.1:9191 >> ./uwsgi.log 2>&1 &
