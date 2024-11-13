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
import csv
import json
import sqlite3
import logging
import subprocess
import click
from math import log
from time import time
from datetime import datetime
from functools import wraps
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from urllib.parse import urlparse, urlencode, urljoin
from flask import (Flask, Markup, Response, jsonify, request, session, g, flash, abort,
		redirect, url_for, render_template, send_file, send_from_directory,
		stream_with_context)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from discodop.tree import (Tree, DrawTree, DrawDependencies, ParentedTree,
		writediscbrackettree)
from discodop.treebank import writetree, writedependencies, exporttree
from discodop.treetransforms import canonicalize
from discodop.treebanktransforms import reversetransform
from discodop.parser import probstr
from discodop.disambiguation import testconstraints
from discodop.heads import applyheadrules
from discodop.eval import editdistance
import worker
from workerattr import workerattr
from activedoptree import ActivedopTree, LABELRE, is_punct_label, cgel_to_ptree
sys.path.append('./cgel')
try:
	import cgel
	from tree2tex import trees2tex
except ImportError:
	cgel = None
	load_as_cgel = None


app = Flask(__name__)  # pylint: disable=invalid-name
# WORKERS = {}  # dict mapping username to process pool
SENTENCES = None
QUEUE = None
ANNOTATIONHELP = """
- If altering the tokenization, ensure that tokens are numbered sequentially.
- Use _. as the token for a gap.
"""
(NBEST, CONSTRAINTS, DECTREE, REATTACH, RELABEL, REPARSE, EDITDIST, TIME
		) = range(8)
# e.g., "NN-SB/Nom" => ('NN', '-SB', '/Nom')

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
		PUNCT_TAGS={}, # dictionary mapping idiosyncratic punctuation POS tags to their corresponding tokens
		SYMBOL_TAG=None, # pos tag for symbols (and symbol sequences) that don't have an idiosyncratic tag (in PUNCT_TAGS)
		AMBIG_SYM={}, # 'ambiguous' symbols that can be punctuation or something else depending on context
		INITIAL_PUNCT_LABELS={}, # pos-function labels for punctuation tokens associated with their following non-punctuation tokens
		WORKERS={},  # dict mapping username to process pool
		)
app.config.from_pyfile('settings.cfg', silent=True)
app.config.from_envvar('FLASK_SETTINGS', silent=True)
WORKERS = app.config['WORKERS']

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

@app.cli.command('initpriorities')
@click.option('--username', default='JoeAnnotator', help='Username to initialize priorities for.')
def initpriorities(username):
	"""Order sentences by entropy of their parse trees probabilities.
	Sentences with saved annotations are included first in the order and are not re-parsed."""
	sentfilename = app.config['SENTENCES']
	db = getdb()
	cur = db.execute(
		'SELECT * FROM entries WHERE username = ? ORDER BY sentno ASC',
		(username, )
	)
	dbentries = cur.fetchall()
	dbentryids = {a[0] for a in dbentries}
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
	already_annotated = []
	for n, entry in enumerate(sentences):
		sent = entry['sentence']
		id = entry['id']
		if id in dbentryids:
			app.logger.info('%d. [already annotated] %s',
					n + 1, sent)
			already_annotated.append((n, 0, sent, id))
		else:
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
	queue = already_annotated + queue
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
	if annotation is not None: # a tree is saved in the database
		# go directly to edit mode
		return redirect(url_for(
				'edit', sentno=sentno, annotated=1, n=n))
	# render annotate mode: browsing parser outputs for the sentence
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
				sent, require, block).result()
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
		prob, ptree, _treestr, _fragments = parsetrees[0]
		treeobj = ActivedopTree(ptree = ptree, senttok = senttok)
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
					tree=treeobj.gtree()))))
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
	parsetrees_ = [(n, prob, ActivedopTree(ptree = ptree, senttok = senttok), treestr, frags)
			for n, (prob, ptree, treestr, frags) in enumerate(parsetrees)
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
					# ad_tree: ActivedopTree object
					tree=ad_tree.gtree())
				for n, prob, ad_tree, _treestr, fragments in parsetrees_)))
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
	senttok, _ = worker.postokenize(sent)
	username = session['username']
	if 'dec' in request.args:
		session['actions'][DECTREE] += int(request.args.get('dec', 0))
	session.modified = True
	msg = ''
	if request.args.get('annotated') == '1': # there is a saved tree
		msg = Markup('<font color=red>You have already annotated '
				'this sentence.</font><button id="undo" onclick="undoAccept()">Delete tree from database</button>')
		id = QUEUE[sentno - 1][3]
		treestr, n = getannotation(username, id) # get tree from database
		treeobj = ActivedopTree.from_str(treestr)
	elif 'n' in request.args: # edit the nth automatic parse
		msg = Markup('<button id="undo" onclick="goback()">Go back</button>')
		n = int(request.args.get('n', 1))
		session['actions'][NBEST] = n
		require = request.args.get('require', '')
		block = request.args.get('block', '')
		require, block = parseconstraints(require, block)
		resp = WORKERS[username].submit(
				worker.getparses,
				sent, require, block).result()
		senttok, parsetrees, _messages, _elapsed = resp
		ptree = parsetrees[n - 1][1]
		treeobj = ActivedopTree(ptree = ptree, senttok = senttok)
	else:
		return 'ERROR: pass n or tree argument.'
	rows = max(5, treeobj.treestr().count('\n') + 1)
	return render_template('edittree.html',
			prevlink=('/annotate/annotate/%d' % (sentno - 1))
				if sentno > 1 else '/annotate/annotate/%d' % (len(SENTENCES)),
			nextlink=('/annotate/annotate/%d' % (sentno + 1))
				if sentno < len(SENTENCES) else '/annotate/annotate/1',
			unextlink=('/annotate/annotate/%d' % firstunannotated(username))
				if sentno < len(SENTENCES) else '#',
			treestr=treeobj.treestr(), senttok=' '.join(senttok), id=id,
			sentno=sentno, lineno=lineno + 1, totalsents=len(SENTENCES),
			numannotated=numannotated(username),
			poslabels=sorted(t for t in workerattr('poslabels') if ('@' not in t) and (t not in app.config['PUNCT_TAGS']) and (t not in app.config['PUNCT_TAGS'].values()) and (t != app.config['SYMBOL_TAG'])),
			phrasallabels=sorted(t for t in workerattr('phrasallabels') if '}' not in t),
			functiontags=sorted(t for t in (workerattr('functiontags')
				| set(app.config['FUNCTIONTAGWHITELIST'])) if '}' not in t and '@' not in t and t != "p"),
			morphtags=sorted(workerattr('morphtags')),
			annotationhelp=ANNOTATIONHELP,
			rows=rows, cols=100,
			msg=msg)

@app.route('/annotate/redraw', methods=['POST'])
@loginrequired
def redraw():
	"""Validate and re-draw tree."""
	data = request.get_json()
	sentno = int(data.get('sentno')) # 1-indexed
	treeobj = ActivedopTree.from_str(data.get('tree'))
	tree_to_accept = treeobj.treestr()
	tree_for_editdist = re.sub(r'\s+', ' ', str(tree_to_accept))
	msg = treeobj.validate()
	link = ('''<a href="#" onclick="accept()">accept this tree</a>
			<input type="hidden" id="sentno" value="%d">'''
		% (sentno))
	oldtree = request.args.get('oldtree', '')
	oldtree = re.sub(r'\s+', ' ', oldtree)
	if oldtree and tree_for_editdist != oldtree:
		session['actions'][EDITDIST] += editdistance(tree_for_editdist, oldtree)
		session.modified = True
	return Markup('%s\n\n%s\n\n%s' % (
			msg,
			link,
			treeobj.gtree(add_editable_attr=True)
			))

def graphical_operation_preamble(treestr):
	treeobj = ActivedopTree.from_str(treestr)
	if app.config['CGELVALIDATE'] is None:
		cgel_tree_terminals = None
	else:
		cgel_tree_terminals = treeobj.cgel_tree.terminals(gaps=True)
	return treeobj, cgel_tree_terminals

def graphical_operation_postamble(dt, senttok, cgel_tree_terminals, sentno):
	ptree = ParentedTree.convert(canonicalize(dt.nodes[0]))
	treeobj = ActivedopTree(ptree = ptree, senttok = senttok, 
						 cgel_tree_terminals = cgel_tree_terminals)
	msg = treeobj.validate()
	link = ('<a href="/annotate/accept?%s">accept this tree</a>'
		% urlencode(dict(sentno=sentno, tree=treeobj.treestr())))	
	return treeobj, link, msg

@app.route('/annotate/newlabel', methods=['POST'])
@loginrequired
def newlabel():
	"""Re-draw tree with newly picked label."""
	data = request.get_json()
	treestr = data.get('tree')
	treeobj, cgel_tree_terminals = graphical_operation_preamble(treestr)
	senttok = treeobj.senttok
	# FIXME: re-factor; check label AFTER replacing it
	# now actually replace label at nodeid
	_treeid, nodeid = data.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	dt = DrawTree(treeobj.ptree, treeobj.senttok)
	m = LABELRE.match(dt.nodes[nodeid].label)
	error = ""
	if data.get('label') is not None:
		label = data.get('label', '')
		dt.nodes[nodeid].label = (label
				+ (m.group(2) or '')
				+ (m.group(3) or ''))
	elif data.get('function') is not None:
		label = data.get('function', '')
		if label == '':
			dt.nodes[nodeid].label = '%s%s' % (
					m.group(1), m.group(3) or '')
		else:
			dt.nodes[nodeid].label = '%s-%s%s' % (
					m.group(1), label, m.group(3) or '')
	elif data.get('morph') is not None:
		label = data.get('morph', '')
		if label == '':
			dt.nodes[nodeid].label = '%s%s' % (
					m.group(1), m.group(2) or '')
		else:
			dt.nodes[nodeid].label = '%s%s/%s' % (
					m.group(1), m.group(2) or '', label)
	else:
		raise ValueError('expected label or function argument')
	treeobj, link, msg = graphical_operation_postamble(dt, senttok, cgel_tree_terminals, int(data.get('sentno'))) 
	if error == '':
		session['actions'][RELABEL] += 1
		session.modified = True
	return Markup('%s\n\n%s\n\n%s%s\t%s' % (
			msg,
			link, error,
			treeobj.gtree(add_editable_attr=True),
			treeobj.treestr()))


@app.route('/annotate/reattach', methods=['POST'])
@loginrequired
def reattach():
	"""Re-draw tree after re-attaching node under new parent."""
	data = request.get_json()
	treestr = data.get('tree')
	treeobj, cgel_tree_terminals = graphical_operation_preamble(treestr)
	dt = DrawTree(treeobj.ptree, treeobj.senttok)
	error = ''
	senttok = treeobj.senttok
	if data.get('newparent') == 'deletenode':
		# remove nodeid by replacing it with its children
		_treeid, nodeid = data.get('nodeid', '').lstrip('t').split('_')
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
	elif data.get('nodeid', '') == 'newproj':
		# splice in a new node under parentid
		_treeid, newparent = data.get('newparent', ''
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
	elif data.get('nodeid', '').startswith('newlabel_'):
		# splice in a new node under parentid
		_treeid, newparent = data.get('newparent', ''
				).lstrip('t').split('_')
		newparent = int(newparent)
		label = data.get('nodeid').split('_', 1)[1]
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
		_treeid, nodeid = data.get('nodeid', '').lstrip('t').split('_')
		nodeid = int(nodeid)
		_treeid, newparent = data.get('newparent', ''
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
							# punctuation positions are subsequently re-canonicalized when ActivedopTree is reconstructed
							node.remove(s)
							dt.nodes[newparent].append(s)
						tree = canonicalize(dt.nodes[0])
						dt = DrawTree(tree, senttok)  # kludge..
					else:
						error = ('ERROR: re-attaching only child creates'
								' empty node %s; remove manually\n' % node)
					break
	treeobj, link, msg = graphical_operation_postamble(dt, senttok, cgel_tree_terminals, int(data.get('sentno'))) 
	if error == '':
		session['actions'][REATTACH] += 1
		session.modified = True
	return Markup('%s\n\n%s\n\n%s%s\t%s' % (
			msg,
			link, error,
			treeobj.gtree(add_editable_attr=True),
			treeobj.treestr()))


@app.route('/annotate/reparsesubtree')
@loginrequired
def reparsesubtree():
	"""Re-parse selected subtree."""
	sentno = int(request.args.get('sentno'))  # 1-indexed
	username = session['username']
	treeobj = ActivedopTree.from_str(request.args.get('tree'))
	dt = DrawTree(treeobj.ptree, treeobj.senttok)
	_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	subseq = sorted(dt.nodes[nodeid].leaves())
	subsent = ' '.join(treeobj.senttok[n] for n in subseq)
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
	username = session['username']
	try:
		treeobj = ActivedopTree.from_str(request.args.get('tree'))
	except ValueError as err:
		return str(err)
	error = ''
	msg = treeobj.validate()
	dt = DrawTree(treeobj.ptree, treeobj.senttok)
	cgel_tree_terminals = treeobj.cgel_tree.terminals(gaps=True)
	_treeid, nodeid = request.args.get('nodeid', '').lstrip('t').split('_')
	nodeid = int(nodeid)
	subseq = sorted(dt.nodes[nodeid].leaves())
	subsent = ' '.join(treeobj.senttok[n] for n in subseq)
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
	ptree = ParentedTree.convert(canonicalize(dt.nodes[0]))
	treeobj = ActivedopTree(ptree = ptree, senttok = treeobj.senttok, 
						 cgel_tree_terminals = cgel_tree_terminals)
	session['actions'][REPARSE] += 1
	session.modified = True
	link = ('<a href="/annotate/accept?%s">accept this tree</a>'
			% urlencode(dict(sentno=sentno, tree=treeobj.treestr())))
	return Markup('%s\n\n%s\n\n%s%s\t%s' % (
			msg,
			link, error,
			treeobj.gtree(add_editable_attr=True),
			treeobj.treestr()))


@app.route('/annotate/accept', methods=['GET', 'POST'])
@loginrequired
def accept():
	"""Store parse & redirect to next sentence."""
	if request.method == 'POST':
		# request.get_json() returns a dictionary
		data = request.get_json()
	elif request.method == 'GET':
		data = request.args
	# should include n referring to which n-best tree is to be accepted,
	# or tree in discbracket format if tree was manually edited.
	sentno = int(data.get('sentno'))  # 1-indexed
	lineno = QUEUE[sentno - 1][0]
	id = QUEUE[sentno - 1][3]
	sent = SENTENCES[lineno]
	username = session['username']
	actions = session['actions']
	actions[TIME] = int(round(time() - actions[TIME]))
	treestr = None
	if 'dec' in data:
		actions[DECTREE] += int(data.get('dec', 0))
	if 'tree' in data:
		n = 0
		treeobj = ActivedopTree.from_str(data.get('tree'))
		tree_to_train = treeobj.ptree
		senttok = treeobj.senttok
		cgel_tree = treeobj.cgel_tree
		# the tokenization may have been updated with gaps, so store the new one
		SENTENCES[lineno] = ' '.join(senttok)
		if False:
			reversetransform(tree, senttok, ('APPEND-FUNC', 'addCase'))
	else:
		n = int(data.get('n', 0))
		require = data.get('require', '')
		block = data.get('block', '')
		require, block = parseconstraints(require, block)
		resp = WORKERS[username].submit(
				worker.getparses,
				sent, require, block).result()
		senttok, parsetrees, _messages, _elapsed = resp
		ptree = parsetrees[n - 1][1]
		treeobj = ActivedopTree(ptree = ptree, senttok = senttok)
		tree_to_train, cgel_tree = treeobj.ptree, treeobj.cgel_tree
		if False:
			# strip function tags
			for node in tree.subtrees():
				node.label = LABELRE.match(node.label).group(1)
	actions[NBEST] = n
	session.modified = True
	block = writetree(tree_to_train.copy(deep=True), senttok, str(lineno + 1), 'export',
		comment='%s %r' % (username, actions))
	app.logger.info(block)
	treeout = block
	addentry(id, lineno, treeout, str(cgel_tree), actions)	# save annotation in the database
	WORKERS[username].submit(worker.augment, [tree_to_train], [senttok])	# update the parser's grammar
	# validate and stay on this sentence if there are issues
	if treestr:
		msg = treeobj.validate()
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

@app.cli.command('cgel2export')
@click.option('--inputfile')
@click.option('--outputfile')
def cgel2export(inputfile, outputfile):
	"""Convert a list of cgel trees from inputfile to Negra export format; write to outputfile
	Produces a list of Negra corpus export-format trees that can be used to train the parser."""
	import copy
	result = []
	with open (inputfile, 'r') as f:
		key = 0
		for tree in cgel.trees(f):
			ptree, senttok = cgel_to_ptree(tree)
			print(tree.metadata)
			treeobj = ActivedopTree(ptree = ptree, senttok = senttok)
			# remove -p function label for training the parser
			for subt in treeobj.ptree.subtrees(lambda t: t.height() == 2):
				if subt.label.endswith("-p"):
					subt.label = subt.label[:-2]
			ptree = treeobj.ptree.copy(deep=True)
			block = writetree(ptree, treeobj.senttok, key=str(key), fmt='export')
			result.append(block)
			key += 1
	with open(outputfile, 'w') as f:
		f.write(''.join(result))
