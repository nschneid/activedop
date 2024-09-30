from collections import defaultdict
import copy
import io
from typing import List
import traceback
from discodop.tree import (Tree, ParentedTree, writediscbrackettree,
							DrawTree, brackettree, discbrackettree)
from discodop.treebank import writetree
import re
import sys
from workerattr import workerattr

sys.path.append('./cgel')
try:
	import cgel
	from scripts.activedopexport2cgel import load as load_as_cgel
except ImportError:
	cgel = None
	load_as_cgel = None

LABELRE = re.compile(r'^([^-/\s]+)(-[^/\s]+)?(/\S+)?$')
PUNCTRE = re.compile(r'^(\W+)$')
COIDXRE = re.compile(r'\.(\w+)')	# coindexation variable in constituent label

# tree functions
ALLOW_EDIT_SENT = True
ALLOW_EDIT_GAPS = True
ALLOW_MULTIWORD_POS = True
ALLOW_UNSEEN_NONCE_CAT = True
ALLOW_UNSEEN_NONCE_FXN = True
ALLOW_UNSEEN_VAR_CAT = True

# punctuation handling functions that may be invoked within app.py

def is_punct_postag(tag):
	from flask import current_app as app
	return tag in app.config['PUNCT_TAGS'].values() or tag == app.config['SYMBOL_TAG']

def is_punct_label(label):
	return label.endswith('-p')

def is_possible_punct_token(token):
	from flask import current_app as app
	return re.match(PUNCTRE, token) or token in app.config['PUNCT_TAGS'] or token in [i['ptree_token'] for i in app.config['PUNCT_ESCAPING']]

class ActivedopTree:
	"""Wrapper for a ParentedTree object with additional methods for activedop."""
	def __init__(self, ptree: ParentedTree, senttok: List[str], cgel_terminals = None):
		"""
		ptree: ParentedTree object with terminals that are numeric indices.
		senttok: list of tokens corresponding to the terminals of ptree.
		cgel_terminals: a list of CGELTree terminals (optional, to replace terminals of CGELTree in initialization).
		"""
		from flask import current_app
		self.app = current_app
		self.ptree = ptree
		self.senttok = senttok
		# standardize ptree with correct labels for punctuation and gaps
		self.ptree = self._apply_standard_labels()
		ptree_terminals_with_labels = copy.deepcopy([subt for subt in self.ptree.subtrees(lambda t: t.height() == 2)])
		# convert ptree to a CGELTree object
		self.cgel_tree = self._ptree_to_cgel()
		# update ptree by canonicalizing the position of punctuation terminals/preterminals
		self.ptree = self._cgel_to_canonicalized_ptree()
		# update canonicalized ptree with standardized labels
		for i, subt in enumerate(self.ptree.subtrees(lambda t: t.height() == 2)):
			subt.label = ptree_terminals_with_labels[i].label
		# update cgel_tree with a set of terminals if provided
		if cgel_terminals is not None:
			self.cgel_tree.update_terminals(cgel_terminals, gaps=True, restore_old_cat=True, restore_old_func=True)
	
	def brackettreestr(self, pretty = False):
		"""returns a string representation of ptree in bracket notation, with labels consisting of a POS tag and a function tag separated by a hyphen."""
		return writediscbrackettree(self.ptree, self.senttok, pretty = pretty)

	def validate(self):
		"""run the brackettree validator on the bracket notation of the tree (plus the CGEL validator if enabled); return the message"""
		_, _, msg = self._validate_disc()
		if self.app.config['CGELVALIDATE'] is not None:
			msg += self._validate_cgel()
		return msg
	
	def treestr(self):
		"""helpful alias for string representation of the tree (CGEL or bracket notation depending on app settings)."""
		if self.app.config['CGELVALIDATE'] is None:
			return self.brackettreestr(pretty=True).rstrip()
		else:
			return str(self.cgel_tree)
	
	def gtree(self, add_editable_attr = False):
		"""returns an html representation of the ptree."""
		out = DrawTree(DrawTree(self.ptree).nodes[0], self.senttok).text(
				unicodelines=True, html=True, funcsep='-',
				morphsep='/', nodeprops='t1', maxwidth=30)
		if add_editable_attr:
			return self._add_editable_attribute(out)
		else:
			return out
		
	def _add_editable_attribute(self, htmltree :str) -> str:
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
		
	def _apply_standard_labels(self) -> ParentedTree:
		"""
		Given a graphical or dopparser-produced tree (punctuation terminals are separate nodes): 
		Enforce consistency of preterminal labels and terminals. 
		(If a punctuation or gap preterminal occurs on the wrong type of terminal, default to N.)
		"""
		# guardrails against producing illict tree structures
		tree, senttok = self.ptree, self.senttok
		tree_copy = tree.copy(deep=True)
		for subt in tree_copy.subtrees(lambda t: t.height() == 2):
			i = subt[0]
			# if initial parse labels non-gaps as GAP, change to N-Head by default
			if subt.label.startswith('GAP') and senttok[i] != '_.':
				subt.label = 'N-Head'
			# condition 1: label consists of a recognized punctuation pos tag, without a function tag [e.g, the label in node `(, ,)`]. Can occur in `annotate` when apply_standard_labels() receives an initial dopparsed tree, and in `edit` when apply_standard_labels() receives a ParentedTree-format PTB-converted ctree. 
			# condition 2: label contains a "p" function tag
			# condition 3: token is unabiguously punctuation
			# -> assign the appropriate punctuation label (either from PUNCT_TAGS or the default SYMBOL_TAG)
			if is_punct_postag(subt.label) or is_punct_label(subt.label) or (is_possible_punct_token(senttok[i]) and senttok[i] not in self.app.config['AMBIG_SYM']):
				subt.label = self.app.config['PUNCT_TAGS'].get(senttok[i], self.app.config['SYMBOL_TAG']) + "-p"
				# if initial parse labels non-punctuation as punctuation, change to N-Head
			if (not is_possible_punct_token(senttok[i])) and (is_punct_label(subt.label)):
				subt.label = 'N-Head'
		return tree_copy
			
	def _ptree_to_cgel(self) -> ParentedTree:
		"""
		Given a graphical or dopparser-produced tree (punctuation terminals are separate nodes): 
		Convert it to a CGELTree object, with prepunct and postpunct attributes assigned to the terminal nodes.
		Assumes that tree has been processed by apply_standard_labels().
		Outputs both a CGELTree object and a ParentedTree object, the latter of which is the cleaned-up version of the input tree with a canonicalized position for punctuation preterminals/terminals.
		"""
		tree, senttok = self.ptree, self.senttok
		tree_copy = tree.copy(deep=True)
		# create three lists of equal lengths: one list non-punctuation token strings, one list of lists prepending punctuation, and one list of lists for appending punctuation
		non_punct_tokens = []
		prepunct_tokens = [[] for subt in tree_copy.subtrees(lambda t: t.height() == 2) if (not is_punct_label(subt.label))]
		postpunct_tokens = copy.deepcopy(prepunct_tokens)

		token_counter = 0

		# iterate through the tree to update the three lists simultaneously
		for subt in tree_copy.subtrees(lambda t: t.height() == 2):
			i = subt[0]
			if subt.label in self.app.config['INITIAL_PUNCT_LABELS'] or (is_punct_label(subt.label) and token_counter == 0):
				prepunct_tokens[token_counter].append(senttok[i])
			elif not is_punct_label(subt.label):
				non_punct_tokens.append(senttok[i])
				token_counter += 1
			elif is_punct_label(subt.label) or token_counter == len(senttok) - 1:
				postpunct_tokens[token_counter - 1].append(senttok[i])

		tree_to_cgel = self._remove_punctuation_nodes(tree_copy) 
		
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
				for e in self.app.config['PUNCT_ESCAPING']:
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
		return cgel_tree

	def _cgel_to_canonicalized_ptree(self) -> ParentedTree:
		"""Convert a cgel_tree to a ParentedTree object. 
		This step canonicalizes the position of punctuation preterminals/terminals."""
		cgel_tree = self.cgel_tree
		treestr = "(ROOT " + cgel_tree.ptb(punct=True, complex_lexeme_separator='_') + ")"
		
		parented_tree, _ = brackettree(treestr)

		return parented_tree
	
	# helper functions for internal _ptree_to_cgel method
	
	def _prune_empty_non_terminals(self, tree: ParentedTree) -> ParentedTree:
		"""Recursively prune empty non-terminal nodes from an NLTK ParentedTree."""
		for i in reversed(range(len(tree))):
			child = tree[i]
			if isinstance(child, ParentedTree):
				pruned_child = self._prune_empty_non_terminals(child)
				if len(pruned_child) == 0:
					del tree[i]
					
		return tree
	
	def _number_terminals(self, tree):
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

	def _remove_punctuation_nodes(self, tree):
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
		return self._number_terminals(self._prune_empty_non_terminals(tree_copy))
	
	# helper functions for validation (used in validate() method)
	
	def _validate_cgel(self):
		cgeltree = self.cgel_tree
		STDERR = sys.stderr
		errS = io.StringIO()
		sys.stderr = errS
		msg = ''
		try:
			nWarn = cgeltree.validate(require_verb_xpos=False, require_num_xpos=False)
		except AssertionError:
			print(traceback.format_exc(), file=errS)
		sys.stderr = STDERR
		if not self.app.config['CGELVALIDATE']:
			msg += '\n(CGEL VALIDATOR IS OFF)\n'
		else:
			errS = errS.getvalue()
			if errS:
				msg += '\nCGEL VALIDATOR\n==============\n' + errS
			else:
				msg += '\nCGEL VALIDATOR: OK\n'
		msg = f'<font color=red>{msg}</font>' if msg else ''
		return msg
	
	def _validate_disc(self):
		treestr, senttok = self.brackettreestr(), self.senttok
		"""Verify whether a user-supplied brackettree (the output of the ActivedopTree.brackettreestr() method) is well-formed."""
		msg = ''
		try:
			tree, sent1 = discbrackettree(treestr)
		except Exception as err:
			raise ValueError('ERROR: cannot parse tree bracketing\n%s' % err)
		# check that sent is not modified
		if senttok!=sent1:
			if [x for x in senttok if not self._isGapToken(x)] == [x for x in sent1 if not self._isGapToken(x)] and ALLOW_EDIT_GAPS:
				# change only to gaps, which is OK
				pass
			elif ALLOW_EDIT_SENT:
				msg += 'Sentence has been modified. '
			else:
				raise ValueError('ERROR: sentence was modified.\n'
						'got:\t%s\nshould be:\t%s' % (
						' '.join(a or '' for a in sent1), ' '.join(senttok)))
		nGaps = len(list(filter(self._isGapToken, sent1)))
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
				if not self._isValidPOS(m.group(1)):
					raise ValueError(('ERROR: invalid POS tag: %s for %d=%s\n'
							'valid POS tags: %s' % (
							node_to_validate.label, node_to_validate[0], senttok[node_to_validate[0]],
							', '.join(sorted(workerattr('poslabels'))))))
				elif m.group(2) and not self._isValidFxn(m.group(2)[1:]):
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
			elif not self._isValidPhraseCat(m.group(1)):
				if ALLOW_UNSEEN_VAR_CAT and '.' in m.group(1):
					msg += f'WARNING: unseen category with variable {m.group(1)} '
				elif ALLOW_UNSEEN_NONCE_CAT and '+' in m.group(1):
					msg += f'WARNING: unseen nonce category {m.group(1)} '
				else:
					raise ValueError(('ERROR: invalid constituent label:\n%s\n'
							'valid labels: %s' % (
							node, ', '.join(sorted(workerattr('phrasallabels'))))))
			if m.group(2) and not self._isValidFxn(m.group(2)[1:]):
				raise ValueError(('ERROR: invalid function tag:\n%s\n'
						'valid labels: %s' % (
						node, ', '.join(sorted(workerattr('functiontags'))))))
		for coindexedset in coindexed.values():
			if len(coindexedset)<2:
				msg += f'ERROR: coindexation variable should have at least two (distinct) constituents: {coindexedset!r} '
				# message not exception because exception blocks display of the tree

		msg = f'<font color=red>{msg}</font>' if msg else ''
		return tree, sent1, msg
	
	# helper functions for validation
	
	def _isGapToken(self, tok):
		return tok.startswith('_.')

	def _isValidPOS(self, x):
		return x in workerattr('poslabels')

	def _isValidPhraseCat(self, x):
		return x in workerattr('phrasallabels') or (ALLOW_MULTIWORD_POS and self._isValidPOS(x))

	def _isValidFxn(self, x):
		return x in workerattr('functiontags') or x in self.app.config['FUNCTIONTAGWHITELIST'] or (ALLOW_UNSEEN_NONCE_FXN and '+' in x)

	@classmethod
	def from_str(cls, tree: str, from_bracket = False, add_root = True):
		"""create an ActivedopTree object from a string representation of a tree (CGEL or bracket notation depending on app settings and `from_bracket` param)."""
		from flask import current_app as app
		if from_bracket or app.config['CGELVALIDATE'] is None:
			if add_root:
				tree = "(ROOT" + tree + ")"
			ptree, senttok = brackettree(tree)
			return cls(ptree, senttok)
		else:
			cgel_tree = cgel.parse(tree)[0]
			cgel_terminals = cgel_tree.terminals(gaps=True)
			tree_bracket = "(ROOT" + cgel_tree.ptb(punct=True) + ")"
			ptree, senttok = brackettree(tree_bracket)
			return cls(ptree, senttok, cgel_terminals)
