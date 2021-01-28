In- and output
--------------
A :class:`Qiber3D.Network` can either be created from from an image stack, or an already reconstructed source like a :file:`.mv3d` or :file:`.swc` file.
To load the network simply pass its path to :meth:`Qiber3D.Network.load`.
Based on the file suffix the corresponding part of :class:`Qiber3D.IO.load` is used.


ND2 example
^^^^^^^^^^^
.. note:: To follow this example you can download the image stack from figshare under `doi:10.6084/m9.figshare.13655606 <https://doi.org/10.6084/m9.figshare.13655606>`_.

.. code-block:: python

    >>> import logging
    >>> from Qiber3D import Network
    >>> from Qiber3D import Network, config
    >>> config.extract.nd2_channel_name = 'FITC'
    >>> config.log_level = logging.DEBUG
    >>> net = Network.load('microvascular_network.nd2')
    Qiber3D_extract [INFO] Load image data from microvascular_network.nd2
    Qiber3D_extract [INFO] Median Filter (despeckle)
    Qiber3D_extract [INFO] Z-Drop correction
    Qiber3D_extract [INFO] Resample image to cubic voxels
    Qiber3D_extract [INFO] Apply gaussian filter
    Qiber3D_extract [INFO] Generate binary representation
    Qiber3D_extract [INFO] Morph binary representation
    Qiber3D_extract [INFO] reconstruct image
    Qiber3D_reconstruct [INFO] Skeletonize image by thinning
    Qiber3D_reconstruct [INFO] Euclidean distance transformation
    Qiber3D_reconstruct [INFO] Link up skeleton
    Qiber3D_reconstruct [INFO] Build Qiber3D.Network for the raw graph
    Qiber3D_reconstruct [INFO] Cleaning Network
    Qiber3D_reconstruct [INFO] Smooth Segments
    >>> print(net)
    Input file: microvascular_network.nd2
      Number of fibers: 459 (clustered 97)
      Number of segments: 660
      Number of branch points: 130
      Total length: 16056.46
      Total volume: 1240236.70
      Average radius: 4.990
      Cylinder radius: 4.959
      Bounding box volume: 681182790

A reconstructed network can quickly be saved with :meth:`Qiber3D.Network.save`.
Without arguments, just the reconstructed network is saved.
If the image stack at the different stages should be preserved set ```save_steps`` to ``True``.
Saving the reconstruction steps can consume a lot of space.

.. code-block:: python

    >>> net.save(save_steps=True)
    Qiber3D_core [INFO] Network saved to Exp190309_PrMECs-NPF180_gel4_ROI-c.qiber

The created :file:`.qiber` can be loaded as any other supported file type.

.. code-block:: python

    >>> net = Network.load('Exp190309_PrMECs-NPF180_gel4_ROI-c.qiber')


Synthetic example
^^^^^^^^^^^^^^^^^

The settings for the different image filters can be accessed through the :mod:`Qiber3D.config` module.
In next example the image to be reconstructed is already cleaned up and just needs to be binarized.
As source we will use a rasterized version of the synthetic test network (:meth:'Qiber3D.IO.load.synthetic_network').

.. code-block:: python

    >>> from Qiber3D import config
    >>> from Qiber3D import Network, IO
    >>> syn_net = IO.load.synthetic_network()
    # set up the resolution of the synthetic network
    >>> voxel_resolution = 5
    >>> config.extract.voxel_size = [1 / voxel_resolution] * 3
    # switching off unneeded filters,
    # as the synthetic network is already cleaned
    >>> config.extract.morph.apply = False
    >>> config.extract.median.apply = False
    >>> config.extract.smooth.apply = False
    >>> config.extract.z_drop.apply = False
    # set a specific threshold to preserve radius
    >>> config.extract.binary.threshold = 29
    >>> config.core_count = 1
    # export the synthetic network as .tif file
    >>> syn_net.export('synthetic.tif', voxel_resolution=voxel_resolution, overwrite=True)
    Qiber3D_render [INFO] Rasterizing network (voxel resolution : 5.00E+00 voxel/unit)
    Qiber3D_core [INFO] Network exported to synthetic.tif.
    # import .tif file
    >>> net = Network.load('synthetic.tif')
    Qiber3D_extract [INFO] Load image data from synthetic.tif
    Qiber3D_extract [INFO] Resample image to cubic voxels
    Qiber3D_extract [INFO] Generate binary representation
    Qiber3D_extract [INFO] reconstruct image
    Qiber3D_reconstruct [INFO] Skeletonize image by thinning
    Qiber3D_reconstruct [INFO] Euclidean distance transformation
    Qiber3D_reconstruct [INFO] Link up skeleton
    Qiber3D_reconstruct [INFO] Build Qiber3D.Network for the raw graph
    Qiber3D_reconstruct [INFO] Cleaning Network
    Qiber3D_reconstruct [INFO] Smooth Segments
    >>> print(net)
    Input file: synthetic.tif
      Number of fibers: 4 (clustered 2)
      Number of segments: 11
      Number of branch points: 4
      Total length: 1120.84
      Total volume: 4665.62
      Average radius: 0.960
      Cylinder radius: 1.151
      Bounding box volume: 800047


Based on the synthetic network we can explore further export options of a :class:`Qiber3D.Network`.

.. code-block:: python

    syn_net.export('synthetic.json')
    Qiber3D_core [INFO] Network exported to synthetic.json

.. literalinclude:: /examples/synthetic.json
   :caption: synthetic.json (truncated)
   :language: json
   :lines: 1-30

.. only:: html

   Full output: :download:`synthetic.json</examples/synthetic.json>`

.. code-block:: python

    syn_net.export('synthetic.xlsx')
    Qiber3D_core [INFO] Network exported to synthetic.xlsx

.. only:: html

   Full output: :download:`synthetic.xlsx</examples/synthetic.xlsx>`



.. code-block:: python

    syn_net.export('synthetic.csv')
    Qiber3D_core [INFO] Network exported to synthetic.csv

.. literalinclude:: /examples/synthetic.csv
   :caption: synthetic.csv (truncated)
   :language: text
   :lines: 1-14

.. only:: html

   Full output: :download:`synthetic.csv</examples/synthetic.csv>`


.. code-block:: python

    syn_net.export('synthetic.mv3d')
    Qiber3D_core [INFO] Network exported to synthetic.mv3d

.. literalinclude:: /examples/synthetic.mv3d
   :caption: synthetic.csv (truncated)
   :language: bash
   :lines: 1-14

.. only:: html

   Full output: :download:`synthetic.mv3d</examples/synthetic.mv3d>`

.. code-block:: python

    syn_net.export('synthetic.x3d')
    Qiber3D_core [INFO] Network exported to synthetic.x3d

.. literalinclude:: /static/x3d/synthetic_show.x3d
   :caption: synthetic.x3d (truncated)
   :language: xml
   :lines: 1-14

.. x3d:: "_static/x3d/synthetic_show.x3d"

.. only:: html

   Full output: :download:`synthetic.x3d</static/x3d/synthetic_show.x3d>`