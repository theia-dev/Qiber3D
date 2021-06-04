Config
--------------
The amount of output can be changed in the config.
Using the `DEBUG` level the results of the extraction steps will be shown.
For a silent operation choose the `ERROR` level.
While using the python debugging module is more verbose, the log_level
can also be set with the corresponding integer values. (`DEBUG`: 10, `INFO`: 20, `WARNING`: 30, `ERROR`: 40)


.. code-block:: python

    >>> import debugging
    >>> from Qiber3D import config
    >>> config.log_level = debugging.DEBUG

    >>> from Qiber3D import Network
    >>> net = Network.load('tests/cases/network_example.tif')


Most settings can be changes in a similar way in the `config`_ module.