from functools import lru_cache
from flask import session
import worker

@lru_cache(maxsize=None, typed=False)
def workerattr(attr):
	from flask import current_app
	WORKERS = current_app.config['WORKERS']
	"""Read attribute of Parser object inside a worker process."""
	username = session['username']
	return WORKERS[username].submit(
			worker.getprop, attr).result()