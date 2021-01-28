Setup
-----

Library
=======

You can install the :mod:`Qiber3D` library directly through the Python Package Index (`PyPI <https://pypi.org>`_).
The use of a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ is recommended.

.. code-block:: shell-session

   $ pip install Qiber3D


If the stable version of Qiber3D on PyPI is missing a particular function, you can install the latest development version directly from the GitHub repository.

.. code-block:: shell-session

    $ pip install -U git+https://github.com/theia-dev/Qiber3D.git#egg=Qiber3D


Development and documentation
=============================

To work with the source code clone the repository from GitHub and install the requirements.
If you add improvements to the code a pull request would be welcomed.
The source code is accompanied by the documentation and a collection of test cases.

.. code-block:: shell-session

    $ git clone https://github.com/theia-dev/Qiber3D.git
    $ python3 -m venv Qiber3D_env
    $ . Qiber3D_env/bin/activate
    (Qiber3D_env) $ pip install --upgrade pip
    (Qiber3D_env) $ pip install -r Qiber3D/requirements.txt

Building the documentation locally needs a few extra python packages.
They can also be installed in the same virtual environment with the following command.

.. code-block:: shell-session

    (Qiber3D_env) $ pip install -r Qiber3D/docs/requirements.txt

The HTML version of the documentation can then be built:

.. code-block:: shell-session

     (Qiber3D_env) $ sphinx-build -b html Qiber3D/docs Qiber3D/docs_html

The tests are located under ``Qiber3D/tests`` can be started with:

.. code-block:: shell-session

      (Qiber3D_env) $ python -m unittest discover -s Qiber3D/tests
