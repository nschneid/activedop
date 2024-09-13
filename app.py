"""Web interface for treebank annotation.

TODO (nschneid):
- validate on accept without entering edit mode
- EVENTUALLY: additional metadata fields e.g. comments, original untokenized sentence

Design notes:

- settings.cfg: The sentences, grammar, user accounts, and other fixed
  configuration parameters are read from a file called "settings.cfg".
  NB: besides the grammar, the treebank the grammar was based on is
  required. Make sure that paths in the grammar parameter file are valid
  (including headrules).
- <filename>.rankings.json: The order in which sentences are annotated
  (prioritization) is fixed before the web service is started, and created by a
  separate command (initpriorities) and stored in a JSON file read at startup
  of the web app.
- annotate.db, schema.sql: Annotations are stored in an sqlite database;
  initialize with "flask initdb".
- session cookie: login status and the per-sentence user interactions are
  stored in a cookie.
- Parsing is done in subprocesses, one for each user, in order to isolate the
  parsing from the Flask process and from other users. Since these subprocesses
  are stored in a global variable (WORKERS), there should only be a single
  Flask process, but multiple threads are needed to get actual parallelism:
  $ flask run --with-threads
  The use of a global variable is a compromise to avoid the complexity of
  running a task queue like celery, or a separate webserver for each user.
"""
import os
import re
import sys
import io
import csv
import json
import sqlite3
import logging
import traceback
import subprocess
import traceback
import copy
import click
from math import log
from time import time
from datetime import datetime
from functools import wraps, lru_cache
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from urllib.parse import urlparse, urlencode, urljoin
from flask import (Flask, Markup, Response, jsonify, request, session, g, flash, abort,
		redirect, url_for, render_template, send_file, send_from_directory,
		stream_with_context)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from discodop.tree import (Tree, ParentedTree, DrawTree, DrawDependencies,
		writediscbrackettree, discbrackettree, brackettree, writebrackettree)
from discodop.treebank import writetree, writedependencies, exporttree
from discodop.treetransforms import canonicalize
from discodop.treebanktransforms import reversetransform
from discodop.parser import probstr
from discodop.disambiguation import testconstraints
from discodop.heads import applyheadrules
from discodop.eval import editdistance
import worker
from pylatexenc.latexencode import unicode_to_latex
from typing import List
sys.path.append('./cgel')
try:
	import cgel
	from scripts.activedopexport2cgel import load as load_as_cgel
	from cgel import Tree as CGELTree
	from tree2tex import trees2tex
except ImportError:
	cgel = None
	load_as_cgel = None


app = Flask(__name__)  # pylint: disable=invalid-name
WORKERS = {}  # dict mapping username to process pool
SENTENCES = None
QUEUE = None
ANNOTATIONHELP = """
- If altering the tokenization, ensure that tokens are numbered sequentially.
- Use _. as the token for a gap.
"""
(NBEST, CONSTRAINTS, DECTREE, REATTACH, RELABEL, REPARSE, EDITDIST, TIME
		) = range(8)
# e.g., "NN-SB/Nom" => ('NN', '-SB', '/Nom')

PUNCT_ESCAPING = [{"istring" : "(", "ptree_label": "LRB-p", "ptree_token": "-LRB-", "ctree_punct": "("},
				  {"istring" : ")", "ptree_label": "RRB-p", "ptree_token": "-RRB-", "ctree_punct": ")"},
				  {"istring" : "-", "ptree_label": "HYPH-p", "ptree_token": "-", "ctree_punct": "-"}]

# punctuation labels based on the Penn Treebank tagset
# [from nltk.help.upenn_tagset()]
PUNCT_TAGS = { 
	# opening parenthesis
	"-LRB-": "LRB", "[" : "LRB", "{": "LRB",
	# closing parenthesis
	"-RRB-": "RRB", "]": "RRB", "}": "RRB",
	# sentence terminator
	".": ".", "!": ".", "?": ".",
	# colon or ellipsis
	":": ":", ";": ":", "...": ":",
	# comma
	",": ",",
	# dash
	"-": "HYPH", "--": "HYPH",
	# opening quotation mark
	"``": "``", "`": "``",
	# closing quotation mark
	"''": "''", "'": "''"
}

# pos tag for symbols (and symbol sequences) that don't have an idiosyncratic PTB tag (in PUNCT_TAGS)
SYMBOL_TAG = "*"

# 'ambiguous' symbols that can be punctuation or something else depending on context
AMBIG_SYM = {
	"$", "#", "%",	# typically N-Head
	"@",	# typically P-Head
	"&", "-", "/"	# typically Coordinator-Marker
}

LABELRE = re.compile(r'^([^-/\s]+)(-[^/\s]+)?(/\S+)?$')
PUNCTRE = re.compile(r'^(\W+)$')
INITIAL_PUNCT_LABELS = {'LRB-p', '[-p', '{-p'}

def is_punct_postag(tag):
	return tag in PUNCT_TAGS.values() or tag == SYMBOL_TAG

def is_punct_label(label):
	return label.endswith('-p')

def is_possible_punct_token(token):
	return re.match(PUNCTRE, token) or token in PUNCT_TAGS or token in [i['ptree_token'] for i in PUNCT_ESCAPING]

def senttok_escape(senttok):
	"""Replace special characters in a tokenized sentence.
	If a token is an 'istring' property of a PUNCT_ESCAPING element, replace it with the 'ptree_token' property."""
	senttok = list(senttok)
	for i, token in enumerate(senttok):
		for e in PUNCT_ESCAPING:
			if token == e['istring']:
				senttok[i] = e['ptree_token']
				break
	return senttok

def sent_escape(sent):
	"""Replace special characters in a sentence. (First splits the sentence into tokens.)
	If a token is an 'istring' property of a PUNCT_ESCAPING element, replace it with the 'ptree_token' property."""
	senttok = sent.split()
	return " ".join(senttok_escape(senttok))

# Load default config and override config from an environment variable
app.config.update(
		DATABASE=os.path.join(app.root_path, 'annotate.db'),
		SECRET_KEY=None,  # set in settings.cfg to protect session cookies
		DEBUG=False,  # whether to enable Flask debug UI
		LIMIT=100,  # maximum sentence length
		FUNCTIONTAGWHITELIST=(),  # optional list of function tags to always accept
		GRAMMAR=None,  # path to a directory with the initial grammar.
		SENTENCES=None,  # a filename with sentences to annotate, one per line.
		ACCOUNTS=None,  # dictionary mapping usernames to passwords
		ANNOTATIONHELP=None,  # plain text file summarizing the annotation scheme
		CGELVALIDATE=None,  # whether to run the CGEL validator when editing
		)
app.config.from_pyfile('settings.cfg', silent=True)
app.config.from_envvar('FLASK_SETTINGS', silent=True)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))


@lru_cache(maxsize=None, typed=False)
def workerattr(attr):
	"""Read attribute of Parser object inside a worker process."""
	username = session['username']
	return WORKERS[username].submit(
			worker.getprop, attr).result()


@app.cli.command('initpriorities')
def initpriorities():
	"""Order sentences by entropy of their parse trees probabilities."""
	sentfilename = app.config['SENTENCES']
	if sentfilename is None:
		raise ValueError('SENTENCES not configured')
	sentences = []
	with open(sentfilename, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)
		for row in csv_reader:
			if any(field.strip() for field in row.values()):
				sentences.append(row)
	# NB: here we do not use a subprocess to do the parsing
	worker.loadgrammar(app.config['GRAMMAR'], app.config['LIMIT'])
	queue = []
	for n, entry in enumerate(sentences):
		sent = entry['sentence']
		id = entry['id']
		try:
			senttok, parsetrees, _messages, _elapsed = worker.getparses(sent)
		except ValueError:
			parsetrees = []
			senttok = []
		app.logger.info('%d. [parse trees=%d] %s',
				n + 1, len(parsetrees), sent)
		ent = 0
		if parsetrees:
			probs = [prob for prob, _tree, _treestr, _deriv in parsetrees]
			try:
				ent = entropy(probs)  # / log(len(parsetrees), 2)
			except (ValueError, ZeroDivisionError):
				pass
		queue.append((n, ent, sent, id))
	queue.sort(key=lambda x: x[1], reverse=True)
	rankingfilename = '%s.rankings.json' % sentfilename
	with open(rankingfilename, 'w') as rankingfile:
		json.dump(queue, rankingfile, indent=4)


@app.before_first_request
def initapp():
	"""Load sentences, check config."""
	global SENTENCES, QUEUE, ANNOTATIONHELP
	sentfilename = app.config['SENTENCES']
	if sentfilename is None:
		raise ValueError('SENTENCES not configured')
	if app.config['GRAMMAR'] is None:
		raise ValueError('GRAMMAR not configured')
	if app.config['ACCOUNTS'] is None:
		raise ValueError('ACCOUNTS not configured')
	# read sentences to annotate
	sentences = []
	with open(sentfilename, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)
		for row in csv_reader:
			if any(field.strip() for field in row.values()):
				sentences.append(row['sentence'])
	SENTENCES = sentences
	rankingfilename = '%s.rankings.json' % sentfilename
	if (os.path.exists(rankingfilename) and
			os.stat(rankingfilename).st_mtime
			> os.stat(sentfilename).st_mtime):
		with open(rankingfilename) as rankingfile:
			QUEUE = json.load(rankingfile)
	else:
		raise ValueError('no rankings for sentences, or sentences have\n'
				'been modified; run "flask initpriorities"')
	if app.config['ANNOTATIONHELP'] is not None:
		with open(app.config['ANNOTATIONHELP']) as inp:
			ANNOTATIONHELP = inp.read()


# Database functions
@app.cli.command('initdb')
def initdb():
	"""Initializes the database."""
	db = getdb()
	with app.open_resource('schema.sql', mode='r') as inp:
		db.cursor().executescript(inp.read())
	db.commit()
	app.logger.info('Initialized the database.')


def connectdb():
	"""Connects to the specific database."""
	result = sqlite3.connect(app.config['DATABASE'])
	# result.row_factory = sqlite3.Row
	return result


def getdb():
	"""Opens a new database connection if there is none yet for the
	current application context."""
	if not hasattr(g, 'sqlitedb'):
		g.sqlitedb = connectdb()
	return g.sqlitedb


@app.teardown_appcontext
def closedb(error):
	"""Closes the database again at the end of the request."""
	if hasattr(g, 'sqlitedb'):
		g.sqlitedb.close()

@app.route('/annotate/get_data_psv')
def get_data_psv():
	username = session['username']
	db = getdb()
	cur = db.execute(
		'SELECT * FROM entries WHERE username = ? ORDER BY sentno ASC',
		(username, )
	)
	rows = cur.fetchall()
	output_dir = "tmp"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)


	csv_file_path = output_dir + '/output.csv'
	with open(csv_file_path, 'w', newline='') as out_file:
		csv_writer = csv.writer(out_file, delimiter='|')
		column_headers = [description[0] for description in cur.description]
		csv_writer.writerow(column_headers)
		for row in rows:
			csv_writer.writerow(row)
	
	return send_file(csv_file_path, as_attachment=True, attachment_filename='data.csv')

def firstunannotated(username):
	"""Return index of first unannotated sentence,
	according to the prioritized order."""
	db = getdb()
	cur = db.execute(
			'select id from entries where username = ? '
			'order by sentno asc',
			(username, ))
	entries = {a[0] for a in cur}
	# sentno=prioritized index, lineno=original index
	for sentno, (_, _, _, id) in enumerate(QUEUE, 1):
		if id not in entries:
			return sentno
	return 1


def numannotated(username):
	"""Number of unannotated sentences for an annotator."""
	db = getdb()
	cur = db.execute('select count(sentno) from entries where username = ?',
			(username, ))
	result = cur.fetchone()
	return result[0]


def getannotation(username, id):
	"""Fetch annotation of a single sentence from database."""
	db = getdb()
	if app.config['CGELVALIDATE'] is None:
		cur = db.execute(
				'select tree, nbest '
				'from entries '
				'where username = ? and id = ? ',
				(username, id))
	else:
		cur = db.execute(
				'select cgel_tree, nbest '
				'from entries '
				'where username = ? and id = ? ',
				(username, id))
	entry = cur.fetchone()
	return (None, 0) if entry is None else (entry[0], entry[1])


def readannotations(username=None):
	"""Get all annotations, or ones by a given annotator."""
	db = getdb()
	if username is None:
		cur = db.execute(
				'select sentno, tree from entries '
				'order by sentno asc')
	else:
		cur = db.execute(
				'select sentno, tree from entries where username = ? '
				'order by sentno asc',
				(username, ))
	entries = cur.fetchall()
	return OrderedDict(entries)


def addentry(id, sentno, tree, cgel_tree, actions):
	"""Add an annotation to the database."""
	db = getdb()
	db.execute(
			'insert or replace into entries '
			'values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
			(id, sentno, session['username'], tree, cgel_tree, *actions,
			datetime.now().strftime('%F %H:%M:%S')))
	db.commit()


def loginrequired(f):
	"""Decorator for views that require a login and running worker pool."""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		if 'username' not in session:
			flash('please log in.')
			return redirect(url_for('login', next=request.url))
		elif session['username'] not in WORKERS:
			return redirect(url_for('dologin', next=request.url))
		return f(*args, **kwargs)
	return decorated_function


def is_safe_url(target):
	"""http://flask.pocoo.org/snippets/62/"""
	ref_url = urlparse(request.host_url)
	test_url = urlparse(urljoin(request.host_url, target))
	return (test_url.scheme in ('http', 'https')
			and ref_url.netloc == test_url.netloc)


# View functions
# NB: when using reverse proxy, the base url should match,
# e.g., https://external.com/annotate/... => http://localhost:5000/annotate/...
@app.route('/')
@app.route('/annotate/')
def main():
	"""Redirect to main page."""
	return redirect(url_for('login'))


@app.route('/annotate/login', methods=['GET', 'POST'])
def login():
	"""Check authentication."""
	error = None
	if request.method == 'POST':
		username = request.form['username']
		if (username not in app.config['ACCOUNTS']
				or request.form['password']
				!= app.config['ACCOUNTS'][username]):
			error = 'Invalid username/password'
		else:  # authentication valid
			session['username'] = username
			if request.args.get('next'):
				return redirect(url_for(
						'dologin', next=request.args.get('next')))
			return redirect(url_for('dologin'))
	elif 'username' in session:
		if request.args.get('next'):
			return redirect(url_for(
					'dologin', next=request.args.get('next')))
		return redirect(url_for('dologin'))
	return render_template(
			'login.html', error=error, totalsents=len(SENTENCES))


@app.route('/annotate/dologin')
def dologin():
	"""Start worker pool and redirect when done."""
	def generate(url):
		yield (
				'<!doctype html>'
				'<title>redirect</title>'
				'You were logged in successfully. ')
		if username in WORKERS:
			try:
				_ = WORKERS[username].submit(
						worker.getprop, 'headrules').result()
			except BrokenProcessPool:
				pass  # fall through
			else:
				yield "<script>window.location.replace('%s');</script>" % url
				return
		yield 'Loading grammar; this will take a few seconds ...'
		_, lang = os.path.split(os.path.basename(app.config['GRAMMAR']))
		app.logger.info('Loading grammar %r', lang)
		pool = ProcessPoolExecutor(max_workers=1)
		if False and app.config['DEBUG']:
			from discodop.treesearch import NoFuture
			future = NoFuture(
					worker.loadgrammar,
					app.config['GRAMMAR'], app.config['LIMIT'])
		else:
			future = pool.submit(
					worker.loadgrammar,
					app.config['GRAMMAR'], app.config['LIMIT'])
		future.result()
		app.logger.info('Grammar %r loaded.', lang)
		# train on annotated sentences
		annotations = readannotations()
		if annotations:
			app.logger.info('training on %d previously annotated sentences',
					len(annotations))
			trees, sents = [], []
			headrules = pool.submit(worker.getprop, 'headrules').result()
			for block in annotations.values():
				# HOTFIX for ROOT error
				blocklns = block.splitlines()
				for iln,blockln in enumerate(blocklns):
					if '\tROOT\t' in blockln and '\t0\t' not in blockln:
						blocklns[iln] = blocklns[iln].replace('\tROOT\t', '\tXXX-XXX\t')
				block = '\n'.join(blocklns)

				item = exporttree(block.splitlines())
				canonicalize(item.tree)
				if headrules:
					applyheadrules(item.tree, headrules)
				trees.append(item.tree)
				sents.append(item.sent)
			if False and app.config['DEBUG']:
				future = NoFuture(worker.augment, trees, sents)
			else:
				future = pool.submit(worker.augment, trees, sents)
			future.result()
		WORKERS[username] = pool
		yield "<script>window.location.replace('%s');</script>" % url
		# return "<script>window.location.replace('%s');</script>" % url

	nexturl = request.args.get('next')
	if not is_safe_url(nexturl) or 'username' not in session:
		return abort(400)
	username = session['username']
	return Response(stream_with_context(generate(
			nexturl or url_for('annotate'))))
	# return generate(nexturl or url_for('annotate'))


# FIXME: add automatic session expiration?
@app.route('/annotate/logout')
def logout():
	"""Log out: clear session, shut down worker pool."""
	if 'username' in session and session['username'] in WORKERS:
		pool = WORKERS.pop(session['username'])
		pool.shutdown(wait=False)
	session.pop('username', None)
	flash('You were logged out')
	return redirect(url_for('main'))


@app.route('/annotate/annotate/', defaults={'sentno': -1})
@app.route('/annotate/annotate/<int:sentno>')
@loginrequired
def annotate(sentno):
	"""Serve the main annotation page for a sentence."""
	username = session['username']
	if sentno == -1:
		sentno = firstunannotated(username)
		redirect(url_for('annotate', sentno=sentno))
	session['actions'] = [0, 0, 0, 0, 0, 0, 0, time()]
	lineno = QUEUE[sentno - 1][0]
	id = QUEUE[sentno - 1][3]
	sent = SENTENCES[lineno]
	senttok, _ = worker.postokenize(sent)
	annotation, n = getannotation(username, id)
	if annotation is not None:
		if app.config['CGELVALIDATE'] is None:
			item = exporttree(annotation.splitlines(), functions='add')
			canonicalize(item.tree)
			worker.domorph(item.tree)
			tree = writediscbrackettree(item.tree, item.sent)
		else: 
			tree = annotation
			_, senttok = brackettree( "(ROOT" + cgel.parse(annotation)[0].ptb(punct = True) + ")" )
		return redirect(url_for(
				'edit', sentno=sentno, annotated=1, tree=tree, n=n))
	return render_template(
			'annotate.html',
			prevlink=str(sentno - 1) if sentno > 1 else str(len(SENTENCES)),
			nextlink=str(sentno + 1) if sentno < len(SENTENCES) else str(1),
			sentno=sentno, lineno=lineno + 1,
			totalsents=len(SENTENCES),
			numannotated=numannotated(username),
			annotationhelp=ANNOTATIONHELP,
			sent=' '.join(senttok))	# includes any gaps

@app.route('/undoaccept', methods=['POST'])
def undoaccept():
	sentid = request.json.get('sentid', 0)
	username = session['username']
	db = getdb()
	db.execute(
	'DELETE FROM entries WHERE username = ? AND id = ?',
	(username, sentid))
	db.commit()
	return jsonify({"success": True})

@app.route('/retokenize', methods=['POST'])
def retokenize():
	sentno = int(request.json.get('sentno', 0))
	newtext = request.json.get('newtext', 0)
	lineno = QUEUE[sentno - 1][0]
	SENTENCES[lineno] = newtext
	return jsonify({"success": True})

@app.route('/annotate/parse')
@loginrequired
def parse():
	"""Display parse. To be invoked by an AJAX call."""
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	sent_esc = sent_escape(sent)
	username = session['username']
	require = request.args.get('require', '')
	block = request.args.get('block', '')
	urlprm = dict(sentno=sentno)
	if require and require != '':
		urlprm['require'] = require
	if block and block != '':
		urlprm['block'] = block
	require, block = parseconstraints(require, block)
	if require or block:
		session['actions'][CONSTRAINTS] += 1
		session.modified = True
	if False and app.config['DEBUG']:
		resp = worker.getparses(sent, require, block)
	else:
		resp = WORKERS[username].submit(
				worker.getparses,
				sent_esc, require, block).result()
	senttok, parsetrees, messages, elapsed = resp
	maxdepth = ''
	if not parsetrees:
		result = ('no parse! reload page to clear constraints, '
				'or continue with next sentence.')
		nbest = dep = depsvg = ''
	else:
		dep = depsvg = ''
		if workerattr('headrules'):
			dep = writedependencies(parsetrees[0][1], senttok, 'conll')
			depsvg = Markup(DrawDependencies.fromconll(dep).svg())
		result = ''
		dectree, maxdepth, _ = decisiontree(parsetrees, senttok, urlprm)
		prob, tree, _treestr, _fragments = parsetrees[0]
		tree_to_viz, _ = tree_process(tree, senttok)
		tree_to_viz = DrawTree(tree_to_viz).nodes[0]
		nbest = Markup('%s\nbest tree: %s' % (
				dectree,
				('%(n)d. [%(prob)s] '
				'<a href="/annotate/accept?%(urlprm)s">accept this tree</a>; '
				'<a href="/annotate/edit?%(urlprm)s">edit</a>; '
				'<a href="/annotate/deriv?%(urlprm)s">derivation</a>\n\n'
				'%(tree)s'
				% dict(
					n=1,
					prob=probstr(prob),
					urlprm=urlencode(dict(urlprm, n=1)),
					tree=DrawTree(tree_to_viz, senttok).text(
						unicodelines=True, html=True, funcsep='-',
						morphsep='/', nodeprops='t1', maxwidth=30)))))
	msg = '\n'.join(messages)
	elapsed = 'CPU time elapsed: %s => %gs' % (
			' '.join('%gs' % a for a in elapsed), sum(elapsed))
	info = '\n'.join((
			'length: %d;' % len(senttok), msg, elapsed,
			'most probable parse trees:',
			''.join('%d. [%s] %s' % (n + 1, probstr(prob),
					writediscbrackettree(treestr, senttok))
					for n, (prob, _tree, treestr, _deriv)
					in enumerate(parsetrees)
					if treestr is not None)
			+ '\n'))
	return render_template('annotatetree.html', sent=sent, result=result,
			nbest=nbest, info=info, dep=dep, depsvg=depsvg, maxdepth=maxdepth,
			msg='%d parse trees' % len(parsetrees))

@app.route('/annotate/filter')
@loginrequired
def filterparsetrees():
	"""For a parse tree in the cache, return a filtered set of its n-best
	parses matching current constraints."""
	username = session['username']
	session['actions'][CONSTRAINTS] += 1
	session.modified = True
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	urlprm = dict(sentno=sentno)
	require = request.args.get('require', '')
	block = request.args.get('block', '')
	if require and require != '':
		urlprm['require'] = require
	if block and block != '':
		urlprm['block'] = block
	require, block = parseconstraints(require, block)
	frequire = request.args.get('frequire', '')
	fblock = request.args.get('fblock', '')
	frequire, fblock = parseconstraints(frequire, fblock)
	resp = WORKERS[username].submit(
			worker.getparses,
			sent, require, block).result()
	senttok, parsetrees, _messages, _elapsed = resp
	parsetrees_ = [(n, prob, tree, treestr, frags)
			for n, (prob, tree, treestr, frags) in enumerate(parsetrees)
			if treestr is None or testconstraints(treestr, frequire, fblock)]
	if len(parsetrees_) == 0:
		return ('No parse trees after filtering; try pressing Re-parse, '
				'or reload page to clear constraints.\n')
	nbest = Markup('%d parse trees\n%s' % (
			len(parsetrees_),
			'\n'.join('%(n)d. [%(prob)s] '
				'<a href="/annotate/accept?%(urlprm)s">accept this tree</a>; '
				'<a href="/annotate/edit?%(urlprm)s">edit</a>; '
				'<a href="/annotate/deriv?%(urlprm)s">derivation</a>\n\n'
				'%(tree)s' % dict(
					n=n + 1,
					prob=probstr(prob),
					urlprm=urlencode(dict(urlprm, n=n + 1)),
					tree=DrawTree(tree, senttok).text(
						unicodelines=True, html=True, funcsep='-', morphsep='/',
						nodeprops='t%d' % (n + 1), maxwidth=30))
				for n, prob, tree, _treestr, fragments in parsetrees_)))
	return nbest


@app.route('/annotate/deriv')
@loginrequired
def showderiv():
	"""Render derivation for a given parse tree in cache."""
	username = session['username']
	n = int(request.args.get('n'))  # 1-indexed
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	require = request.args.get('require', '')
	block = request.args.get('block', '')
	require, block = parseconstraints(require, block)
	resp = WORKERS[username].submit(
			worker.getparses,
			sent, require, block).result()
	senttok, parsetrees, _messages, _elapsed = resp
	_prob, tree, _treestr, fragments = parsetrees[n - 1]
	return Markup(
			'<pre>Fragments used in the highest ranked derivation'
			' of this parse tree:\n%s\n%s</pre>' % (
			'\n\n'.join(
				'%s\n%s' % (w, DrawTree(frag).text(unicodelines=True, html=True))
				for frag, w in fragments or ()),
			DrawTree(tree, senttok).text(
				unicodelines=True, html=True, funcsep='-', maxwidth=30)))


@app.route('/annotate/edit')
@loginrequired
def edit():
	"""Edit tree manually."""
	sentno = int(request.args.get('sentno'))  # 1-indexed
	lineno = QUEUE[sentno - 1][0]
	id = QUEUE[sentno - 1][3]
	sent = SENTENCES[lineno]
	sent_esc = sent_escape(sent)
	senttok, _ = worker.postokenize(sent)
	username = session['username']
	if 'dec' in request.args:
		session['actions'][DECTREE] += int(request.args.get('dec', 0))
	session.modified = True
	msg = ''
	if request.args.get('annotated', False):
		msg = Markup('<font color=red>You have already annotated '
				'this sentence.</font><button id="undo" onclick="undoAccept()">Delete tree from database</button>')
		if app.config['CGELVALIDATE'] is None:
			tree, senttok = discbrackettree(request.args.get('tree'))
		else:
			cgel_tree = cgel.parse(request.args.get('tree'))[0]
			tree = "(ROOT" + cgel_tree.ptb(punct = True) + ")"
			tree, senttok = brackettree(tree)
			tree = writediscbrackettree(tree,senttok)
			treestr = cgel_tree
	elif 'n' in request.args:
		msg = Markup('<button id="undo" onclick="goback()">Go back</button>')
		n = int(request.args.get('n', 1))
		session['actions'][NBEST] = n
		require = request.args.get('require', '')
		block = request.args.get('block', '')
		require, block = parseconstraints(require, block)
		resp = WORKERS[username].submit(
				worker.getparses,
				sent_esc, require, block).result()
		senttok, parsetrees, _messages, _elapsed = resp
		tree = parsetrees[n - 1][1]
		if app.config['CGELVALIDATE'] is None:
			pass
		else:
			_, cgel_tree = tree_process(tree, senttok)
			treestr = cgel_tree
	elif 'tree' in request.args:
		msg = Markup('<button id="undo" onclick="goback()">Go back</button>')
		if app.config['CGELVALIDATE'] is None:
			tree, senttok = discbrackettree(request.args.get('tree'))
		else:
			cgel_tree = cgel.parse(request.args.get('tree'))[0]
			tree = "(ROOT" + cgel_tree.ptb(punct = True) + ")"
			tree, senttok = brackettree(tree)
			tree = writediscbrackettree(tree,senttok)
			treestr = cgel_tree
	else:
		return 'ERROR: pass n or tree argument.'
	if app.config['CGELVALIDATE'] is None:
		treestr = writediscbrackettree(tree, senttok, pretty=True).rstrip()
		rows = max(5, treestr.count('\n') + 1)
	else:
		rows = max(5, treestr.depth)
	return render_template('edittree.html',
			prevlink=('/annotate/annotate/%d' % (sentno - 1))
				if sentno > 1 else '/annotate/annotate/%d' % (len(SENTENCES)),
			nextlink=('/annotate/annotate/%d' % (sentno + 1))
				if sentno < len(SENTENCES) else '/annotate/annotate/1',
			unextlink=('/annotate/annotate/%d' % firstunannotated(username))
				if sentno < len(SENTENCES) else '#',
			treestr=treestr, senttok=sent, id=id,
			sentno=sentno, lineno=lineno + 1, totalsents=len(SENTENCES),
			numannotated=numannotated(username),
			poslabels=sorted(t for t in workerattr('poslabels') if ('@' not in t) and (t not in PUNCT_TAGS.values()) and (t != SYMBOL_TAG)),
			phrasallabels=sorted(t for t in workerattr('phrasallabels') if '}' not in t),
			functiontags=sorted(t for t in (workerattr('functiontags')
				| set(app.config['FUNCTIONTAGWHITELIST'])) if '}' not in t and '@' not in t and t != "p"),
			morphtags=sorted(workerattr('morphtags')),
			annotationhelp=ANNOTATIONHELP,
			rows=rows, cols=100,
			msg=msg)

def prune_empty_non_terminals(tree: ParentedTree) -> ParentedTree:
	# Recursively prune children first
	for i in reversed(range(len(tree))):
		child = tree[i]
		if isinstance(child, ParentedTree):
			pruned_child = prune_empty_non_terminals(child)
			if len(pruned_child) == 0:
				del tree[i]
				
	return tree

def number_terminals(tree):
	"""
	Number the terminal nodes in a ParentedTree sequentially starting from 0.

	Args:
	tree (ParentedTree): The ParentedTree to renumber terminal nodes.

	Returns:
	ParentedTree: The updated ParentedTree with numbered terminal nodes.
	"""
	terminal_count = 0  # Initialize the terminal counter

	def _number_terminals(node):
		nonlocal terminal_count
		if isinstance(node, ParentedTree):
			for i, child in enumerate(node):
				if isinstance(child, ParentedTree):
					_number_terminals(child)
				else:
					# Assign a new terminal number
					node[i] = terminal_count
					terminal_count += 1

	# Create a copy of the tree to avoid modifying the original
	tree_copy = tree.copy(deep=True)
	_number_terminals(tree_copy)
	return tree_copy

def remove_punctuation_nodes(tree):
	"""
	Recursively remove punctuation nodes from an NLTK ParentedTree.

	Args:
	tree (ParentedTree): The tree from which to remove punctuation nodes.

	Returns:
	ParentedTree: The tree with punctuation nodes removed.
	"""

	# Traverse the tree and remove punctuation nodes
	def _remove_punct(tree):
		if isinstance(tree, ParentedTree):
			children_to_remove = []
			for i, child in enumerate(tree):
				if isinstance(child, ParentedTree):
					if is_punct_label(child.label):
						children_to_remove.append(i)
					else:
						_remove_punct(child)

			# Remove children from the tree after collecting indices
			for i in reversed(children_to_remove):
				del tree[i]

	# Create a copy of the tree to avoid modifying the original
	tree_copy = tree.copy(deep=True)
	_remove_punct(tree_copy)
	return number_terminals(prune_empty_non_terminals(tree_copy))

def tree_process(tree : ParentedTree, senttok: List[str]) -> tuple[ParentedTree, CGELTree]:
	"""
	Given a graphical or dopparser-produced tree (punctuation terminals are separate nodes): 
	1. Clean it up to canonicalize the position of punctuation, and to enforce consistency of 
	preterminals and terminals. (If a punctuation or gap preterminal occurs on the wrong type 
	of terminal, default to N.)
	2. Create a CGELTree object from the cleaned tree, with prepunct and postpunct attributes 
	assigned to the terminal nodes.
	"""
	# guardrails against producing illict tree structures
	tree_copy = tree.copy(deep=True)
	ptree_terminals = []
	for subt in tree_copy.subtrees(lambda t: t.height() == 2):
		i = subt[0]
		# if initial parse labels non-gaps as GAP, change to N-Head by default
		if subt.label.startswith('GAP') and senttok[i] != '_.':
			subt.label = 'N-Head'
		# condition 1: label consists of a recognized punctuation pos tag, without a function tag [e.g, the label in node `(, ,)`]. Can occur in `annotate` when tree_process() receives an initial dopparsed tree, and in `edit` when tree_process() receives a ParentedTree-format PTB-converted ctree. 
		# condition 2: label contains a "p" function tag
		# condition 3: token is unabiguously punctuation
		# -> assign the appropriate punctuation label (either from PUNCT_TAGS or the default SYMBOL_TAG)
		if is_punct_postag(subt.label) or is_punct_label(subt.label) or (is_possible_punct_token(senttok[i]) and senttok[i] not in AMBIG_SYM):
			subt.label = PUNCT_TAGS.get(senttok[i], SYMBOL_TAG) + "-p"
			# if initial parse labels non-punctuation as punctuation, change to N-Head
		if (not is_possible_punct_token(senttok[i])) and (is_punct_label(subt.label)):
			subt.label = 'N-Head'
		ptree_terminals.append(subt)

	# create three lists of equal lengths: one list non-punctuation token strings, one list of lists prepending punctuation, and one list of lists for appending punctuation
	non_punct_tokens = []
	prepunct_tokens = [[] for subt in tree_copy.subtrees(lambda t: t.height() == 2) if (not is_punct_label(subt.label))]
	postpunct_tokens = copy.deepcopy(prepunct_tokens)

	token_counter = 0

	# iterate through the tree to update the three lists simultaneously
	for subt in tree_copy.subtrees(lambda t: t.height() == 2):
		i = subt[0]
		if subt.label in INITIAL_PUNCT_LABELS or (is_punct_label(subt.label) and token_counter == 0):
			prepunct_tokens[token_counter].append(senttok[i])
		elif not is_punct_label(subt.label):
			non_punct_tokens.append(senttok[i])
			token_counter += 1
		elif is_punct_label(subt.label) or token_counter == len(senttok) - 1:
			postpunct_tokens[token_counter - 1].append(senttok[i])

	tree_to_cgel = remove_punctuation_nodes(tree_copy) 
	
	try:
		block = writetree(tree_to_cgel, non_punct_tokens, '1', 'export', comment='')
		block = io.StringIO(block)
		cgel_tree = next(load_as_cgel(block))
	except AssertionError:
		_, _, tb = sys.exc_info()
		traceback.print_tb(tb)
		tb_info = traceback.extract_tb(tb)
		# if we get this error, it means that ROOT has multiple children, which is not allowed by the CGEL parser.
		# fallback strategy: place contents of ROOT node under a new node labeled 'Clause'
		if tb_info[1].line == 'assert root is None':
			# if some subtree is called ROOT (or Clause w/o function), change to Clause-Head by default
			for subt in tree_to_cgel.subtrees():
				if subt.label == 'ROOT':
					subt.label = 'Clause-Head'
				elif "-" not in subt.label:
					subt.label = subt.label + "-Head"
			tree_to_cgel.label = 'Clause'
			tree_to_cgel = ParentedTree('ROOT', [tree_to_cgel])
			block = writetree(tree_to_cgel, non_punct_tokens, '1', 'export', comment='')
			block = io.StringIO(block)
			cgel_tree = next(load_as_cgel(block))

	cgel_tree_terminals = cgel_tree.terminals(gaps=True)

	def unescape_ptree_tok(token_list):
		for i, p in enumerate(token_list):
			for e in PUNCT_ESCAPING:
				if p == e['ptree_token']:
					token_list[i] = e['ctree_punct']
					break

	for i, terminal in enumerate(cgel_tree_terminals):

		prepunct_token_list = prepunct_tokens[i]
		unescape_ptree_tok(prepunct_token_list)
		postpunct_token_list = postpunct_tokens[i]
		unescape_ptree_tok(postpunct_token_list)
		terminal.prepunct = prepunct_token_list
		terminal.postpunct = postpunct_token_list
		if terminal.text:
			terminal.text = terminal.text.replace("_", " ")

	cgel_tree.update_terminals(cgel_tree_terminals, gaps=True)

	treestr = "(ROOT " + cgel_tree.ptb(punct=True, complex_lexeme_separator='_') + ")"
	
	parented_tree, _ = brackettree(treestr)

	for i, subt in enumerate(parented_tree.subtrees(lambda t: t.height() == 2)):
		subt.label = ptree_terminals[i].label

	return (parented_tree, cgel_tree)

def add_editable_attribute(htmltree :str) -> str:
	""" 
	Given an html rendering of a tree [the output of DrawTree(... html=True ...) or DrawTree.text(... html=True ...)], output an html tree 
	in which tree preterminals (span elements of class 'p') and function tags (span elements of class 'f') have a feature called 'editable'.
	This feature determines whether the user is able to change function/category labels of preterminals on the graphical tree. 
	Editability is turned off for preterminals that include punctuation function/pos tags. 
	(Has to use regex because beautifulsoup wrecks the tree formatting.)
	"""
	# add editable attribute to non-punctuation preterminals
	htmltree_preterminals = re.findall(r'<span\s+class=p[^>]*>', htmltree)
	for preterminal in htmltree_preterminals:
		# extract the preterminal's function and pos tag from the span's `data-s` attribute:
		label = re.search(r'data-s="([^"]*)"', preterminal).group(1).split(' ')[0]
		m = LABELRE.match(label)
		if m.group(2) == "-p" or is_punct_postag(m.group(1)):
			htmltree = htmltree.replace(preterminal, preterminal.replace('class=p', 'class=p editable="false"'))
		else:
			htmltree = htmltree.replace(preterminal, preterminal.replace('class=p', 'class=p editable="true"'))
	# add editable attribute to non-punctuation function tags
	htmltree_functiontags = re.findall(r'<span\s+class=f[^>]*>', htmltree)
	for functiontag in htmltree_functiontags:
		# extract the function tag from the span's `data-s` attribute:
		label = re.search(r'data-s="([^"]*)"', functiontag).group(1)
		if is_punct_label(label):
			htmltree = htmltree.replace(functiontag, functiontag.replace('class=f', 'class=f editable="false"'))
		else:
			htmltree = htmltree.replace(functiontag, functiontag.replace('class=f', 'class=f editable="true"'))
	return htmltree
	
@app.route('/annotate/redraw')
@loginrequired
def redraw():
	"""Validate and re-draw tree."""
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	orig_senttok, _ = worker.postokenize(sent)
	if app.config['CGELVALIDATE'] is None:
		treestr = request.args.get('tree')
		senttok = orig_senttok
		tree_for_editdist = re.sub(r'\s+', ' ', treestr)
		tree_to_validate = treestr
	else: 
		treestr = request.args.get('tree')
		cgel_tree = cgel.parse(treestr)[0]
		tree_to_viz, senttok = brackettree("(ROOT " + cgel_tree.ptb(punct=True, complex_lexeme_separator='_') + ")")
		tree_to_viz, _ = tree_process(tree_to_viz, senttok)
		tree_for_editdist = re.sub(r'\s+', ' ', str(cgel_tree))
		tree_to_validate = writediscbrackettree(DrawTree(tree_to_viz).nodes[0],senttok)
	try:
		tree_to_viz, senttok, msg = validate(tree_to_validate, senttok)
		if app.config['CGELVALIDATE'] is not None:
			msg += validate_cgel(cgel_tree)
	except ValueError as err:
		return str(err)
	link = ('<a href="/annotate/accept?%s">accept this tree</a>'
		% urlencode(dict(sentno=sentno, tree=treestr)))
	oldtree = request.args.get('oldtree', '')
	oldtree = re.sub(r'\s+', ' ', oldtree)
	if oldtree and tree_for_editdist != oldtree:
		session['actions'][EDITDIST] += editdistance(tree_for_editdist, oldtree)
		session.modified = True
	return Markup('%s\n\n%s\n\n%s' % (
			msg,
			link,
			add_editable_attribute(DrawTree(tree_to_viz, senttok).text(
				unicodelines=True, html=True, funcsep='-', morphsep='/',
				nodeprops='t0', maxwidth=30))
			))

def graphical_operation_preamble():
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	sent_esc = sent_escape(sent)
	orig_senttok, _ = worker.postokenize(sent_esc)
	if app.config['CGELVALIDATE'] is None:
		treestr = request.args.get('tree')
		cgel_tree_terminals = None
		senttok = orig_senttok
	else:
		cgel_tree = cgel.parse(request.args.get('tree'))[0]
		cgel_tree_terminals = cgel_tree.terminals(gaps=True)
		ptb_tree = "(ROOT " + cgel_tree.ptb(punct=True, complex_lexeme_separator='_') + ")"
		tree, senttok = brackettree(ptb_tree)
		tree, _ = tree_process(tree, senttok)
		treestr = writediscbrackettree(DrawTree(tree).nodes[0],senttok)
	msg = ""
	return tree, senttok, msg, treestr, orig_senttok, cgel_tree_terminals, sentno

def graphical_operation_postamble(dt, senttok, cgel_tree_terminals, orig_senttok, sentno):
	tree = dt.nodes[0]
	tree = brackettree(writediscbrackettree(tree, senttok))[0]
	senttok = senttok_escape(senttok)
	if app.config['CGELVALIDATE'] is None:
		tree_to_viz, _ = tree_process(tree, senttok)
		dt = DrawTree(tree_to_viz, senttok) # kludge..
		treestr = writediscbrackettree(tree, senttok, pretty=True).rstrip()
		# validate after operation
		_, _, msg = validate(treestr, senttok, cgel_validate=False)
	else:
		tree_to_viz, cgel_tree = tree_process(tree, senttok)
		cgel_tree.update_terminals(cgel_tree_terminals, gaps=True, restore_old_cat=True, restore_old_func=True)
		treestr = cgel_tree
		msg = validate_cgel(cgel_tree)
		dt = DrawTree(tree_to_viz, senttok)
	link = ('<a href="/annotate/accept?%s">accept this tree</a>'
		% urlencode(dict(sentno=sentno, tree=treestr)))
	return treestr, dt, link, msg

@app.route('/annotate/newlabel')
@loginrequired
def newlabel():
	"""Re-draw tree with newly picked label."""
	tree, senttok, msg, treestr, orig_senttok, cgel_tree_terminals, sentno = graphical_operation_preamble()
	# FIXME: re-factor; check label AFTER replacing it
	# now actually replace label at nodeid
	_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	dt = DrawTree(tree, senttok)
	m = LABELRE.match(dt.nodes[nodeid].label)
	error = ""
	if 'label' in request.args:
		label = request.args.get('label', '')
		dt.nodes[nodeid].label = (label
				+ (m.group(2) or '')
				+ (m.group(3) or ''))
	elif 'function' in request.args:
		label = request.args.get('function', '')
		if label == '':
			dt.nodes[nodeid].label = '%s%s' % (
					m.group(1), m.group(3) or '')
		else:
			dt.nodes[nodeid].label = '%s-%s%s' % (
					m.group(1), label, m.group(3) or '')
	elif 'morph' in request.args:
		label = request.args.get('morph', '')
		if label == '':
			dt.nodes[nodeid].label = '%s%s' % (
					m.group(1), m.group(2) or '')
		else:
			dt.nodes[nodeid].label = '%s%s/%s' % (
					m.group(1), m.group(2) or '', label)
	else:
		raise ValueError('expected label or function argument')
	treestr, dt, link, msg = graphical_operation_postamble(dt, senttok, cgel_tree_terminals, orig_senttok, sentno)
	if error == '':
		session['actions'][RELABEL] += 1
		session.modified = True
	return Markup('%s\n\n%s\n\n%s%s\t%s' % (
			msg,
			link, error,
			add_editable_attribute(dt.text(unicodelines=True, html=True, funcsep='-', morphsep='/',
				nodeprops='t0', maxwidth=30)),
			treestr))


@app.route('/annotate/reattach')
@loginrequired
def reattach():
	"""Re-draw tree after re-attaching node under new parent."""
	tree, senttok, msg, treestr, orig_senttok, cgel_tree_terminals, sentno = graphical_operation_preamble()
	dt = DrawTree(tree, senttok)
	error = ''
	if request.args.get('newparent') == 'deletenode':
		# remove nodeid by replacing it with its children
		_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
		nodeid = int(nodeid)
		x = dt.nodes[nodeid]
		if nodeid == 0 or isinstance(x[0], int):
			error = 'ERROR: cannot remove ROOT or POS node'
		else:
			children = list(x)
			x[:] = []
			for y in dt.nodes[0].subtrees():
				if any(child is x for child in y):
					i = y.index(x)
					y[i:i + 1] = children
					tree = canonicalize(dt.nodes[0])
					dt = DrawTree(tree, senttok)  # kludge..
					break
	elif request.args.get('nodeid', '') == 'newproj':
		# splice in a new node under parentid
		_treeid, newparent = request.args.get('newparent', ''
				).lstrip('t').split('_')
		newparent = int(newparent)
		y = dt.nodes[newparent]
		label = y.label
		if isinstance(y[0], int):
			error = 'ERROR: cannot add node under POS tag'
		else:
			children = list(y)
			y[:] = []
			y[:] = [Tree(label, children)]
			tree = canonicalize(dt.nodes[0])
			dt = DrawTree(tree, senttok)  # kludge..
	elif request.args.get('nodeid', '').startswith('newlabel_'):
		# splice in a new node under parentid
		_treeid, newparent = request.args.get('newparent', ''
				).lstrip('t').split('_')
		newparent = int(newparent)
		label = request.args.get('nodeid').split('_', 1)[1]
		y = dt.nodes[newparent]
		if isinstance(y[0], int):
			error = 'ERROR: cannot add node under POS tag'
		else:
			children = list(y)
			y[:] = []
			y[:] = [Tree(label, children)]
			tree = canonicalize(dt.nodes[0])
			dt = DrawTree(tree, senttok)  # kludge..
	else:  # re-attach existing node at existing new parent
		_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
		nodeid = int(nodeid)
		_treeid, newparent = request.args.get('newparent', ''
				).lstrip('t').split('_')
		newparent = int(newparent)
		# remove node from old parent
		# dt.nodes[nodeid].parent.pop(dt.nodes[nodeid].parent_index)
		x = dt.nodes[nodeid]
		y = dt.nodes[newparent]

		def find_self_and_sisters(tree, subtree):
			parent = None
			sisters = []

			# Helper function to find the parent of the subtree
			def find_parent(node, target):
				nonlocal parent
				if target in node.children:
					parent = node
					return True
				for child in node.children:
					if isinstance(child, int):
						return False
					elif find_parent(child, target):
						return True
				return False

			# Find the parent of the subtree
			find_parent(tree, subtree)

			if parent:
				# Collect all children of the parent node
				sisters = [child for child in parent.children]

			return sisters
		
		def extract_adjacent_punctuation(arr, target):
			# Find the index of the target character
			try:
				target_index = arr.index(target)
			except ValueError:
				return []  # If target is not in the list, return an empty list
	
			# Initialize the result list with the target character
			result = [target]

			# Collect punctuation characters to the left of the target
			left_index = target_index - 1
			while left_index >= 0 and is_punct_label(arr[left_index].label):
				result.insert(0, arr[left_index])
				left_index -= 1
			
			# Collect punctuation characters to the right of the target
			right_index = target_index + 1
			while right_index < len(arr) and is_punct_label(arr[right_index].label):
				result.append(arr[right_index])
				right_index += 1
	
			return result
		
		for node in x.subtrees():
			if node is y:
				error = ('ERROR: cannot re-attach subtree'
						' under (descendant of) itself\n')
				break
		else:
			for node in dt.nodes[0].subtrees():
				if any(child is x for child in node):
					if len(node) > 1:
						self_and_sisters = find_self_and_sisters(dt.nodes[0], x)
						self_and_nearbypunct = extract_adjacent_punctuation(self_and_sisters, x)
						for s in self_and_nearbypunct:
							# iteratively move all sister punctuation to the target. 
							# (prevents problematic crossover movement of non-punctuation nodes over punctuation nodes)
							# punctuation positions are subsequently re-canonicalized with a call to tree_process()
							node.remove(s)
							dt.nodes[newparent].append(s)
						tree = canonicalize(dt.nodes[0])
						dt = DrawTree(tree, senttok)  # kludge..
					else:
						error = ('ERROR: re-attaching only child creates'
								' empty node %s; remove manually\n' % node)
					break
	treestr, dt, link, msg = graphical_operation_postamble(dt, senttok, cgel_tree_terminals, orig_senttok, sentno)
	if error == '':
		session['actions'][REATTACH] += 1
		session.modified = True
	return Markup('%s\n\n%s\n\n%s%s\t%s' % (
			msg,
			link, error,
			add_editable_attribute(dt.text(unicodelines=True, html=True, funcsep='-', morphsep='/',
				nodeprops='t0', maxwidth=30)),
			treestr))


@app.route('/annotate/reparsesubtree')
@loginrequired
def reparsesubtree():
	"""Re-parse selected subtree."""
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	orig_senttok, _ = worker.postokenize(sent)
	username = session['username']
	treestr = request.args.get('tree', '')
	try:
		tree, senttok, msg = validate(treestr, orig_senttok)
	except ValueError as err:
		return str(err)
	error = ''
	dt = DrawTree(tree, senttok)
	_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	subseq = sorted(dt.nodes[nodeid].leaves())
	subsent = ' '.join(senttok[n] for n in subseq)
	# FIXME only works when root label of tree matches label in grammar.
	# need a single label that works across all stages.
	root = dt.nodes[nodeid].label
	# root = grammar.tolabel[next(iter(grammar.tblabelmapping[root]))]
	resp = WORKERS[username].submit(
			worker.getparses,
			subsent,
			(), (),
			root=root).result()
	_senttok, parsetrees, _messages, _elapsed = resp
	app.logger.info('%d-%d. [parse trees=%d] %s',
			sentno, nodeid, len(parsetrees), subsent)
	print(parsetrees[0][1])
	nbest = Markup('<pre>%d parse trees\n'
			'<a href="javascript: toggle(\'nbest\'); ">cancel</a>\n'
			'%s</pre>' % (
			len(parsetrees),
			'\n'.join('%(n)d. [%(prob)s] '
				'<a href="#" onClick="picksubtree(%(n)d); ">'
				'use this subtree</a>; '
				'\n\n'
				'%(tree)s' % dict(
					n=n + 1,
					prob=probstr(prob),
					tree=DrawTree(tree, subsent.split()).text(
						unicodelines=True, html=True, funcsep='-',
						morphsep='/', nodeprops='t%d' % (n + 1), maxwidth=30))
				for n, (prob, tree, _treestr, fragments)
				in enumerate(parsetrees))))
	return nbest


@app.route('/annotate/replacesubtree')
@loginrequired
def replacesubtree():
	n = int(request.args.get('n', 0))
	sentno = int(request.args.get('sentno'))  # 1-indexed
	sent = SENTENCES[QUEUE[sentno - 1][0]]
	orig_senttok, _ = worker.postokenize(sent)
	username = session['username']
	treestr = request.args.get('tree')
	if app.config['CGELVALIDATE'] is None:
		treestr = request.args.get('tree')
	else: 
		treestr = "(ROOT " + cgel.parse(request.args.get('tree'))[0].ptb(punct=False, complex_lexeme_separator='_') + ")"
		treestr = writediscbrackettree(DrawTree(treestr).nodes[0],orig_senttok)
	try:
		tree, senttok, msg = validate(treestr, orig_senttok)
	except ValueError as err:
		return str(err)
	error = ''
	dt = DrawTree(tree, senttok)
	_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	subseq = sorted(dt.nodes[nodeid].leaves())
	subsent = ' '.join(senttok[n] for n in subseq)
	root = dt.nodes[nodeid].label
	resp = WORKERS[username].submit(
			worker.getparses,
			subsent, (), (),
			root=root).result()
	_senttok, parsetrees, _messages, _elapsed = resp
	newsubtree = parsetrees[n - 1][1]
	pos = sorted(list(newsubtree.subtrees(
			lambda n: isinstance(n[0], int))),
			key=lambda n: n[0])
	for n, a in enumerate(pos):
		a[0] = subseq[n]
	dt.nodes[nodeid][:] = newsubtree[:]
	tree = canonicalize(dt.nodes[0])
	dt = DrawTree(tree, senttok)  # kludge..
	treestr = writediscbrackettree(tree, senttok, pretty=True).rstrip()
	session['actions'][REPARSE] += 1
	session.modified = True
	link = ('<a href="/annotate/accept?%s">accept this tree</a>'
			% urlencode(dict(sentno=sentno, tree=treestr)))
	return Markup('%s\n\n%s\n\n%s%s\t%s' % (
			msg,
			link, error,
			dt.text(unicodelines=True, html=True, funcsep='-', morphsep='/',
				nodeprops='t0', maxwidth=30),
			treestr))


@app.route('/annotate/accept')
@loginrequired
def accept():
	"""Store parse & redirect to next sentence."""
	# should include n referring to which n-best tree is to be accepted,
	# or tree in discbracket format if tree was manually edited.
	sentno = int(request.args.get('sentno'))  # 1-indexed
	lineno = QUEUE[sentno - 1][0]
	id = QUEUE[sentno - 1][3]
	sent = SENTENCES[lineno]
	username = session['username']
	actions = session['actions']
	actions[TIME] = int(round(time() - actions[TIME]))
	treestr = None
	if 'dec' in request.args:
		actions[DECTREE] += int(request.args.get('dec', 0))
	if 'tree' in request.args:
		n = 0
		if app.config['CGELVALIDATE'] is None:
			treestr = request.args.get('tree')
			tree_to_train, senttok = discbrackettree(treestr)
			cgel_tree = "none"
		else:
			cgel_tree = cgel.parse(request.args.get('tree'))[0]
			tree_to_train, senttok = brackettree(cgel_tree.ptb(punct=True, complex_lexeme_separator='_'))
			tree_to_train, _ = tree_process(tree_to_train, senttok)
		# the tokenization may have been updated with gaps, so store the new one
		SENTENCES[lineno] = ' '.join(senttok)
		if False:
			reversetransform(tree, senttok, ('APPEND-FUNC', 'addCase'))
	else:
		n = int(request.args.get('n', 0))
		require = request.args.get('require', '')
		block = request.args.get('block', '')
		require, block = parseconstraints(require, block)
		resp = WORKERS[username].submit(
				worker.getparses,
				sent, require, block).result()
		senttok, parsetrees, _messages, _elapsed = resp
		tree = parsetrees[n - 1][1]
		tree_to_train, cgel_tree = tree_process(tree, senttok)
		if False:
			# strip function tags
			for node in tree.subtrees():
				node.label = LABELRE.match(node.label).group(1)
	actions[NBEST] = n
	session.modified = True
	block = writetree(tree_to_train, senttok, str(lineno + 1), 'export',
		comment='%s %r' % (username, actions))
	app.logger.info(block)
	treeout = block
	addentry(id, lineno, treeout, str(cgel_tree), actions)	# save annotation in the database
	WORKERS[username].submit(worker.augment, [tree_to_train], [senttok])	# update the parser's grammar
	# validate and stay on this sentence if there are issues
	if treestr:
		_tree, _senttok, msg = validate(treestr, senttok)
		if 'ERROR' in msg or 'WARNING' in msg:
			flash('Your annotation for sentence %d was stored %r but may contain errors. Please click Validate to check.' % (sentno, actions))
			return redirect(url_for('annotate', sentno=sentno))
	flash('Your annotation for sentence %d was stored %r' % (sentno, actions))
	return (redirect(url_for('annotate', sentno=1))
		if sentno >= len(SENTENCES)
		else redirect(url_for('annotate', sentno=sentno+1)))

@app.route('/annotate/context/<int:lineno>')
def context(lineno):
	"""Show all sentences, in original order."""
	ranking = {a[0]: n for n, a in enumerate(QUEUE, 1)}
	return render_template('context.html',
			sentences=SENTENCES, ranking=ranking, lineno=lineno)


@app.route('/annotate/export')
def export():
	"""Export annotations by current user."""
	return Response(
			''.join(readannotations(session['username']).values()),
			mimetype='text/plain')

@app.route('/annotate/exportallcgeltrees')
def exportallcgeltrees():
	"""Export all annotations by current user in .cgel format."""
	assert load_as_cgel
	username = session['username']
	db = getdb()
	cur = db.execute(
			'select cgel_tree from entries where username = ? '
			'order by sentno asc',
			(username, ))
	entries = [a[0] for a in cur]
	cgeltrees = '\n'.join(entries)
	return Response(cgeltrees, mimetype='text/plain')

@app.route('/annotate/download_pdf')
def download_pdf():
	cgeltree = cgel.parse(request.args.get('tree'))
	cgel_latex = trees2tex(cgeltree)
	output_dir = "tmp"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(os.path.join(output_dir, "file.tex"), 'w') as latex_file:
		latex_file.write(cgel_latex)

	subprocess.run(['pdflatex', '-output-directory', output_dir, os.path.join(output_dir, "file.tex")])

	pdf_path = os.path.join(output_dir, "file.pdf")

	return send_file(pdf_path, as_attachment=True, attachment_filename='downloaded_file.pdf')

@app.route('/annotate/exportcgeltree')
def exportcgeltree():
	"""Produce single tree in .cgel format"""
	cgeltree = request.args.get('tree')
	return Response(cgeltree, mimetype='text/plain')

@app.route('/annotate/favicon.ico')
@app.route('/favicon.ico')
def favicon():
	"""Serve the favicon."""
	return send_from_directory(os.path.join(app.root_path, 'static'),
			'parse.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/annotate/static/script.js')
def javascript():
	"""Serve javascript."""
	return send_from_directory(os.path.join(app.root_path, 'static'),
			'script.js', mimetype='text/javascript')


@app.route('/annotate/static/style.css')
def stylecss():
	"""Serve style.css."""
	return send_from_directory(os.path.join(app.root_path, 'static'),
			'style.css', mimetype='text/css')

def isGapToken(tok):
	return tok.startswith('_.')

# tree functions
ALLOW_EDIT_SENT = True
ALLOW_EDIT_GAPS = True
ALLOW_MULTIWORD_POS = True
ALLOW_UNSEEN_NONCE_CAT = True
ALLOW_UNSEEN_NONCE_FXN = True
ALLOW_UNSEEN_VAR_CAT = True
COIDXRE = re.compile(r'\.(\w+)')	# coindexation variable in constituent label

def isValidPOS(x):
	return x in workerattr('poslabels')

def isValidPhraseCat(x):
	return x in workerattr('phrasallabels') or (ALLOW_MULTIWORD_POS and isValidPOS(x))

def isValidFxn(x):
	return x in workerattr('functiontags') or x in app.config['FUNCTIONTAGWHITELIST'] or (ALLOW_UNSEEN_NONCE_FXN and '+' in x)

def validate_cgel(cgeltree):
	STDERR = sys.stderr
	errS = io.StringIO()
	sys.stderr = errS
	msg = ''
	try:
		nWarn = cgeltree.validate(require_verb_xpos=False, require_num_xpos=False)
	except AssertionError:
		print(traceback.format_exc(), file=errS)
	sys.stderr = STDERR
	if not app.config['CGELVALIDATE']:
		msg += '\n(CGEL VALIDATOR IS OFF)\n'
	else:
		errS = errS.getvalue()
		if errS:
			msg += '\nCGEL VALIDATOR\n==============\n' + errS
		else:
			msg += '\nCGEL VALIDATOR: OK\n'
	msg = f'<font color=red>{msg}</font>' if msg else ''
	return msg

def validate(treestr, senttok):
	"""Verify whether a user-supplied tree is well-formed."""
	msg = ''
	try:
		tree, sent1 = discbrackettree(treestr)
	except Exception as err:
		raise ValueError('ERROR: cannot parse tree bracketing\n%s' % err)
	# check that sent is not modified
	if senttok!=sent1:
		if [x for x in senttok if not isGapToken(x)] == [x for x in sent1 if not isGapToken(x)] and ALLOW_EDIT_GAPS:
			# change only to gaps, which is OK
			pass
		elif ALLOW_EDIT_SENT:
			msg += 'Sentence has been modified. '
		else:
			raise ValueError('ERROR: sentence was modified.\n'
					'got:\t%s\nshould be:\t%s' % (
					' '.join(a or '' for a in sent1), ' '.join(senttok)))
	nGaps = len(list(filter(isGapToken, sent1)))
	if nGaps>0:
		msg += f'Sentence contains {nGaps} gap(s). '
	# check tree structure
	coindexed = defaultdict(set)	# {coindexationvar -> {labels}}
	for node in tree.subtrees():
		if node is not tree.root and node.label==tree.root.label:
			raise ValueError(('ERROR: non-root node cannot have same label as root: '+node.label))
		m = LABELRE.match(node.label)
		if m is None:
			raise ValueError('malformed label: %r\n'
					'expected: cat-func/morph or cat-func; e.g. NN-SB/Nom'
					% node.label)
		else:
			mCoidx = COIDXRE.search(node.label)
			if mCoidx:
				coindexed[mCoidx.group(1)].add(node.label)
		if len(node) == 0:
			raise ValueError(('ERROR: a constituent should have '
					'one or more children:\n%s' % node))
		# create copy of node to validate POS and function tags (stripping -p from label if present)
		node_to_validate = copy.deepcopy(node)
		if node_to_validate.label.endswith('-p'):
			node_to_validate.label = node.label[:-2]
		# a POS tag
		elif isinstance(node_to_validate[0], int):
			if not isValidPOS(m.group(1)):
				raise ValueError(('ERROR: invalid POS tag: %s for %d=%s\n'
						'valid POS tags: %s' % (
						node_to_validate.label, node_to_validate[0], senttok[node_to_validate[0]],
						', '.join(sorted(workerattr('poslabels'))))))
			elif m.group(2) and not isValidFxn(m.group(2)[1:]):
				raise ValueError(('ERROR: invalid function tag:\n%s\n'
						'valid labels: %s' % (
						node, ', '.join(sorted(workerattr('functiontags'))))))
			elif len(node) != 1:
				raise ValueError(('ERROR: a POS tag must have exactly one '
						'token as child and nothing else:\n%s' % node))
		# not a POS tag but a phrasal node
		elif not all(isinstance(child, Tree) for child in node):
			raise ValueError(('ERROR: a constituent cannot have a token '
					'as child:\n%s' % node))
		elif not isValidPhraseCat(m.group(1)):
			if ALLOW_UNSEEN_VAR_CAT and '.' in m.group(1):
				msg += f'WARNING: unseen category with variable {m.group(1)} '
			elif ALLOW_UNSEEN_NONCE_CAT and '+' in m.group(1):
				msg += f'WARNING: unseen nonce category {m.group(1)} '
			else:
				raise ValueError(('ERROR: invalid constituent label:\n%s\n'
						'valid labels: %s' % (
						node, ', '.join(sorted(workerattr('phrasallabels'))))))
		if m.group(2) and not isValidFxn(m.group(2)[1:]):
			raise ValueError(('ERROR: invalid function tag:\n%s\n'
					'valid labels: %s' % (
					node, ', '.join(sorted(workerattr('functiontags'))))))
	for coindexedset in coindexed.values():
		if len(coindexedset)<2:
			msg += f'ERROR: coindexation variable should have at least two (distinct) constituents: {coindexedset!r} '
			# message not exception because exception blocks display of the tree

	msg = f'<font color=red>{msg}</font>' if msg else ''
	return tree, sent1, msg

def entropy(seq):
	"""Calculate entropy of a probability distribution.

	>>> entropy([0.25, 0.25, 0.25, 0.25])  # high uncertainty, high entropy
	2.0
	>>> entropy([0.9, 0.05, 0.05])  # low uncertainty, low entropy
	0.5689955935892812
	"""
	if not seq:
		return 0
	probmass = sum(seq)
	probs = [prob / probmass for prob in seq]
	return -sum(p * log(p, 2) for p in probs)


def parseconstraints(require, block):
	"""
	>>> parseconstraints("NP 0-2\tPP 0-1,4", "")
	(('NP', [0, 1, 2]), ('PP', [0, 1, 4])), ()
	"""
	def constr(item):
		"""Parse a single constraint."""
		label, span = item.split(' ', 1)
		seq = []
		for rng in span.split(','):
			if '-' in rng:
				b, c = rng.split('-')
				seq.extend(range(int(b), int(c) + 1))
			else:
				seq.append(int(rng))
		return label, seq

	if require:
		require = tuple((label, tuple(indices))
				for label, indices in sorted(map(constr, require.split('\t'))))
	else:
		require = ()
	if block:
		block = tuple((label, tuple(indices))
				for label, indices in sorted(map(constr, block.split('\t'))))
	else:
		block = ()
	return require, block


def getspans(tree):
	"""Yield spans of Tree object."""
	for node in tree.subtrees():
		if node is not tree:  # skip root
			yield node.label, tuple(sorted(node.leaves()))


def decisiontree(parsetrees, sent, urlprm):
	"""Create a decision tree to select among n trees."""
	# The class labels are the n-best trees 0..n
	# The attributes are the labeled spans in the trees; they split the n-best
	# trees into two sets with and without that span.
	spans = {}
	if len(parsetrees) <= 1:
		return '', 0, None
	for n, (_prob, tree, _, _) in enumerate(parsetrees):
		for span in getspans(tree):
			# simplest strategy: store presence of span as binary feature
			# perhaps better: use weight from tree probability
			spans.setdefault(span, set()).add(n)

	# create decision tree with scikit-learn
	features = list(spans)
	featurenames = ['[%s %s]' % (label, ' '.join(sent[n] for n in leaves))
			for label, leaves in features]
	data = np.array([[n in spans[span] for span in features]
			for n in range(len(parsetrees))], dtype=bool)
	estimator = DecisionTreeClassifier(random_state=0)
	estimator.fit(data, range(len(parsetrees)),
			sample_weight=[prob for prob, _, _, _ in parsetrees])
	path = estimator.decision_path(data)

	def rec(tree, n=0, depth=0):
		"""Recursively produce a string representation of a decision tree."""
		if tree.children_left[n] == tree.children_right[n]:
			x = tree.value[n].nonzero()[1][0]
			prob, _tree, _treestr, _fragments = parsetrees[x]
			thistree = ('%(n)d. [%(prob)s] '
					'<a href="/annotate/accept?%(urlprm)s">accept this tree</a>; '
					'<a href="/annotate/edit?%(urlprm)s">edit</a>; '
					'<a href="/annotate/deriv?%(urlprm)s">derivation</a>\n\n'
					% dict(
						n=x + 1,
						prob=probstr(prob),
						urlprm=urlencode(dict(urlprm, n=x + 1, dec=depth))))
			return ('<span id="d%d" style="display: none; ">%stree %d:\n'
					'%s</span>' % (n, depth * '\t', x + 1, thistree))
		left = tree.children_left[n]
		right = tree.children_right[n]
		return ('<span id=d%(n)d style="display: %(display)s; ">'
				'%(indent)s%(constituent)s '
				'<a href="javascript: showhide(\'d%(right)s\', \'d%(left)s\', '
					'\'dd%(exright)s\', \'%(numtrees)s\'); ">'
					'good constituent</a> '
				'<a href="javascript: showhide(\'d%(left)s\', \'d%(right)s\', '
					'\'dd%(exleft)s\', \'%(numtrees)s\'); ">'
					'bad constituent</a> '
				'%(subtree1)s%(subtree2)s</span>' % dict(
				n=n,
				display='block' if n == 0 else 'none',
				indent=depth * 4 * ' ',
				constituent=featurenames[tree.feature[n]],
				left=left, right=right,
				exleft=path[:, left].nonzero()[0][0],
				exright=path[:, right].nonzero()[0][0],
				numtrees=len(parsetrees),
				subtree1=rec(tree, left, depth + 1),
				subtree2=rec(tree, right, depth + 1),
				))
	nodes = rec(estimator.tree_)
	leaves = []
	seen = set()
	for n in range(estimator.tree_.node_count):
		x = estimator.tree_.value[n].nonzero()[1][0]
		if x in seen:
			continue
		seen.add(x)
		_prob, xtree, _treestr, _fragments = parsetrees[x]
		thistree = DrawTree(xtree, sent).text(
				unicodelines=True, html=True, funcsep='-', morphsep='/',
				nodeprops='t%d' % (x + 1), maxwidth=30)
		leaves.append('<span id="dd%d" style="display: none; ">%s</span>' %
				(x, thistree))
	return nodes + ''.join(leaves), estimator.tree_.max_depth, path

if __name__ == '__main__':
	pass

@app.cli.command('ptb2ptree')
@click.option('--inputfile')
@click.option('--outputfile')
def ptb2ptree(inputfile, outputfile):
	"""Convert a list of ptb-labelled bracketed trees from inputfile to ptree-labelled bracketed trees; write to outputfile
	Produces a list of trees that can be used to train the parser."""
	result = []
	with open (inputfile, 'r') as f:
		for line in f:
			ptree, senttok = brackettree("(ROOT " + line.strip() + ")")
			# escape punctuation preterminals
			for subt in ptree.subtrees(lambda t: t.height() == 2):
				if subt.label in PUNCT_TAGS:
					subt.label = PUNCT_TAGS[subt.label]
				elif is_possible_punct_token(subt.label):
					subt.label = SYMBOL_TAG
			ptree, _ = tree_process(ptree, senttok)
			# remove -p function label for training the parser
			for subt in ptree.subtrees(lambda t: t.height() == 2):
				if subt.label.endswith("-p"):
					subt.label = subt.label[:-2]
			ptree = writebrackettree(ptree[0], senttok).rstrip()
			result.append(ptree)
	with open(outputfile, 'w') as f:
		f.write('\n'.join(result))
