Active DOP treebank annotation tool
===================================

A treebank annotation tool based on a statistical parser that is re-trained during annotation.
Paper: http://www.aclweb.org/anthology/C18-2009

.. image:: screenshot.png
   :alt: screenshot of annotation tool

Installation instructions (MacOS and Linux)
-------------------------------------------

Requires Python 3.10-11 to install. 

0. (Recommended): Create and activate a venv virtual Python environment:

.. code-block:: bash

   python3.11 -m venv .venv
   . .venv/bin/activate

1. Install submodule requirements:

.. code-block:: bash

   pip install setuptools
   pip install cython==3.0.12

2. Install submodules:

.. code-block:: bash

   git submodule update --init --recursive
   cd roaringbitmap
   python setup.py install
   cd ..
   cd disco-dop
   pip3 install -r requirements.txt
   env CC=gcc sudo python setup.py install
   cd ..

3. Install activedop:

.. code-block:: bash

   pip3 install -r requirements.txt

Installation instructions (Windows PowerShell)
----------------------------------------------

NOTE: Requires a C++ compiler, e.g., from Visual Studio Build Tools. 

Make sure to also install a Windows 10/11 SDK.

You will also need a standard GCC distribution to compile discodop. These instructions assume you've installed GCC from https://www.msys2.org/.

0. (Recommended): Create and activate a venv virtual Python environment:

.. code-block:: powershell

   python3 -m venv .venv
   .\.venv\Scripts\activate

1. Install submodule requirements:

.. code-block:: powershell

   pip install setuptools
   pip install cython

2. Apply patches to dependencies 

Installing discodop on Windows requires a few patches. 

First, in :code:`disco-dop/setup.py`, add :code:`'-DMS_WIN64'` to the array of :code:`extra_compile_args` at line 111.
Then, in the same file, redefine :code:`extra_link_args` at line 128 such that the line reads:

.. code-block:: python

   extra_link_args = ['-DNDEBUG', '-static-libgcc', '-static-libstdc++', '-Wl,-Bstatic,--whole-archive', '-lwinpthread', '-Wl,--no-whole-archive']

3. Install submodules:

.. code-block:: powershell

   git submodule update --init --recursive
   cd .\roaringbitmap\
   python setup.py install
   cd ..
   cd .\disco-dop\
   pip3 install -r requirements.txt
   python setup.py build --compiler=mingw32
   python setup.py install
   cd ..

4. Install activedop:

.. code-block:: powershell

   pip3 install -r requirements.txt

Running the demo on a toy treebank and annotation task:
-------------------------------------------------------

- extract the example grammar: "discodop runexp example.prm"
  The grammar will be extracted from "treebankExample.mrg",
  and the annotation task will consist of the sentences in "newsentsExample.txt".
- run "FLASK_APP=app.py flask initdb"
- run "FLASK_APP=app.py flask initpriorities"
- start the web server with "FLASK_APP=app.py flask run --with-threads".
  open browser at http://localhost:5000/
  username "JoeAnnotator", password "example"

Configuration
-------------

Edit "settings.cfg" to use a different grammar and sentences to annotate,
and to configure usernames and passwords.
Note that the treebank on which the grammar is based needs to be available,
in the paths specified in the grammar parameter file.

Annotators have the option to export the annotated trees as LaTeX/PDF files.
To use a local LaTeX installation for PDF generation, set the
`LATEX_SERVICE` configuration option to "local".
To use a remote LaTeX service, set `LATEX_SERVICE` to "remote", and set the
`LATEX_SERVICE_URL` and `LATEX_CREDS` environment variables accordingly.

The remote LaTeX service should accept POST requests with JSON payloads of the form

.. code-block:: bash
   {
      "latex_code": "<latex code>",
   }

and return the generated PDF as binary content.

The `LATEX_CREDS` value should be a string of the form "username:password",
to be used for HTTP Basic Authentication with the remote LaTeX service.

The local LaTeX installation needs to have the `pdflatex` command available in the system PATH.

Input sentences
----------------

Sentences need to be segmented, one sentence per line. For best results,
tokenize the sentences to annotate according to treebank conventions.


Reference
---------
bibtex::

    @InProceedings{vancranenburgh2018active,
        author={van Cranenburgh, Andreas},
        title={Active DOP: A constituency treebank annotation tool with online learning}
        year={2018},
        booktitle={Proceedings of COLING system demonstrations},
        pages={38--42},
        url={http://www.aclweb.org/anthology/C18-2009}
    }

