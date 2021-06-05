Logging
--------------
The amount of output can be changed.
Using the `DEBUG` level the results of the extraction steps will be shown.
For a silent operation choose the `ERROR` level.
While using the python debugging module is more verbose, the log_level
can also be set with the corresponding integer values. (`DEBUG`: 10, `INFO`: 20, `WARNING`: 30, `ERROR`: 40)


.. code-block:: python

    >>> import debugging
    >>> from Qiber3D import Network, helper
    >>> helper.change_log_level(logging.DEBUG)

    >>> net = Network.load('tests/cases/network_example.tif')

