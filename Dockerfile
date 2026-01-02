FROM python:3.11
EXPOSE 5000/tcp
WORKDIR /app

# Install dependencies and submodules

COPY requirements.txt requirements.txt
COPY roaringbitmap roaringbitmap/
COPY disco-dop/ disco-dop/

RUN pip3 install setuptools
RUN pip3 install cython==3.0.12

# OPTIONAL: Install LaTeX for PDF generation (if not using an external service)
# RUN apt-get update && apt-get install -y make texlive-latex-extra

WORKDIR roaringbitmap/
RUN python setup.py install
WORKDIR ../disco-dop/
RUN pip3 install -r requirements.txt
RUN env CC=gcc python setup.py install
WORKDIR ..

RUN pip3 install -r requirements.txt

# Copy the rest of the resources

COPY templates/ templates/
COPY cgel/ cgel/
COPY cgelbank2-punct/ cgelbank2-punct/
COPY static/ static/

COPY settings.cfg settings.cfg

# Assumes a local SQLite database named annotate.db exists
COPY annotate.db annotate.db

COPY app.py app.py
COPY activedoptree.py activedoptree.py
COPY schema.sql schema.sql
COPY worker.py worker.py
COPY workerattr.py workerattr.py

COPY newsentsExample.csv newsentsExample.csv
COPY newsentsExample.csv.rankings.json newsentsExample.csv.rankings.json

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run the Flask application with threading enabled
CMD ["python", "-m", "flask", "run", "--with-threads"]