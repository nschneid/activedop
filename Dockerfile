FROM python:3.12.3
EXPOSE 5000/tcp
WORKDIR /app

# Copy resources

COPY roaringbitmap roaringbitmap/
COPY templates/ templates/
COPY cgel/ cgel/
COPY cgelbank2-punct/ cgelbank2-punct/
COPY disco-dop/ disco-dop/
COPY static/ static/

COPY requirements.txt requirements.txt
COPY settings.cfg settings.cfg
COPY annotate.db annotate.db

COPY app.py app.py
COPY activedoptree.py activedoptree.py
COPY schema.sql schema.sql
COPY worker.py worker.py
COPY workerattr.py workerattr.py

COPY newsentsExample.csv newsentsExample.csv
COPY newsentsExample.csv.rankings.json newsentsExample.csv.rankings.json

# Install dependencies

RUN apt-get update && apt-get install -y make texlive-latex-extra && \
	pip3 install cython && \
	pip3 install setuptools
WORKDIR roaringbitmap/
RUN python setup.py install
WORKDIR ../disco-dop/
RUN pip3 install -r requirements.txt
RUN env CC=gcc python setup.py install
WORKDIR ..
RUN pip3 install -r requirements.txt

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run the Flask application with threading enabled
CMD ["python", "-m", "flask", "run", "--with-threads"]