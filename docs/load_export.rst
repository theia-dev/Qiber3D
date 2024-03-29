In- and output
--------------
A :class:`Qiber3D.Network` can either be created from from an image stack, or an already reconstructed source like a :file:`.mv3d` or :file:`.swc` file.
To load the network simply pass its path to :meth:`Qiber3D.Network.load`.
Based on the file suffix the corresponding part of :class:`Qiber3D.IO.load` is used.

.. note:: The examples presented here use the corresponding image stacks from figshare
          doi:`10.6084/m9.figshare.13655606 <https://doi.org/10.6084/m9.figshare.13655606>`_.
          The :class:`Qiber3D.helper.Example` class can be used to download the data easily.

          * :file:`microvascular_network.nd2` - FigID: 26211077 (1.06 GB)
          * :file:`microvascular_network.tif` - FigID: 30771817 (1.06 GB)
          * :file:`microvascular_network-C2.tif` - FigID: 30771877 (362 MB)
          * :file:`microvascular_network-C2-reduced.tif` - FigID: 31106104 (40MB)

.. note:: To directly download the files on the commandline ``curl`` can be used.
          Just replace ``$(FigID)`` with the number from the list above.

          ``curl -X GET "https://api.figshare.com/v2/file/download/$(FigID)"``


ND2 example
^^^^^^^^^^^

.. code-block:: python

    >>> import logging
    >>> from Qiber3D import Network, config
    >>> from Qiber3D.helper import Example, change_log_level
    >>> config.log_level = change_log_level(logging.DEBUG)  # This will show 3D renderings of the intermediate steps
    >>> config.extract.nd2_channel_name = 'FITC'
    >>> net_ex = Example.nd2()
    >>> net = Network.load(net_ex)
    Qiber3D_extract [INFO] Load image data from microvascular_network.nd2
    Qiber3D_extract [INFO] Image voxel size: [1.230,1.230,2.500]
    Qiber3D_extract [INFO] Median Filter (despeckle)
    Qiber3D_extract [INFO] Z-Drop correction
    Qiber3D_extract [INFO] Resample image to cubic voxels
    Qiber3D_extract [INFO] Apply gaussian filter
    Qiber3D_extract [INFO] Generate binary representation
    Qiber3D_extract [INFO] Binary representation used a threshold of 4.9% (otsu)
    Qiber3D_extract [INFO] Morph binary representation
    Qiber3D_extract [INFO] reconstruct image
    Qiber3D_reconstruct [INFO] Skeletonize image by thinning
    Qiber3D_reconstruct [INFO] Euclidean distance transformation
    Link up skeleton: 100%|██████████| 13/13 [00:15<00:00,  1.17s/it]
    Qiber3D_reconstruct [INFO] Build Qiber3D.Network from the raw graph
    Qiber3D_reconstruct [INFO] Cleaning Network
    Qiber3D_reconstruct [INFO] Smooth Segments
    >>> print(net)
    Input file: microvascular_network.nd2
      Number of fibers: 405 (clustered 109)
      Number of segments: 966
      Number of branch points: 327
      Total length: 43192.45
      Total volume: 5146410.65
      Average radius: 5.770
      Cylinder radius: 6.158
      Bounding box volume: 683663872



A reconstructed network can quickly be saved with :meth:`Qiber3D.Network.save`.
Without arguments, just the reconstructed network is saved.
If the image stack at the different stages should be preserved set ```save_steps`` to ``True``.
Saving the reconstruction steps can consume a lot of space.

.. code-block:: python

    >>> net.save(save_steps=True)
    Qiber3D_core [INFO] Network saved to microvascular_network.qiber

The created :file:`.qiber` can be loaded as any other supported file type.

.. code-block:: python

    >>> net = Network.load('microvascular_network.qiber')


TIFF example
^^^^^^^^^^^^

The same example can be run using a `.tif` version of the same data.

.. note:: While the reduced dataset is the same subject, the reduced resolution and greyscale range will result in altered results.


.. code-block:: python

    >>> from Qiber3D import Network, config
    >>> from Qiber3D.helper import Example
    >>> config.extract.voxel_size = [1.2302, 1.2302, 2.5]
    >>> config.extract.low_memory = True
    >>> net_ex = Example.tiff()
    >>> net = Network.load(net_ex, channel=1)
    Qiber3D_extract [INFO] Load image data from microvascular_network.tif
    Qiber3D_extract [INFO] Image voxel size: [1.230,1.230,2.500]
    Qiber3D_extract [INFO] Median Filter (despeckle)
    Qiber3D_extract [INFO] Z-Drop correction
    Qiber3D_extract [INFO] Resample image to cubic voxels
    Qiber3D_extract [INFO] Apply gaussian filter
    Qiber3D_extract [INFO] Generate binary representation
    Qiber3D_extract [INFO] Binary representation used a threshold of 4.9% (otsu)
    Qiber3D_extract [INFO] Morph binary representation
    Qiber3D_extract [INFO] reconstruct image
    Qiber3D_reconstruct [INFO] Skeletonize image by thinning
    Qiber3D_reconstruct [INFO] Euclidean distance transformation
    Qiber3D_reconstruct [INFO] Build Qiber3D.Network from the raw graph
    Qiber3D_reconstruct [INFO] Cleaning Network
    Qiber3D_reconstruct [INFO] Smooth Segments
    >>> print(net)
    Input file: microvascular_network.tif
      Number of fibers: 406 (clustered 111)
      Number of segments: 971
      Number of branch points: 329
      Total length: 43241.93
      Total volume: 5150823.10
      Average radius: 5.775
      Cylinder radius: 6.158
      Bounding box volume: 683658224

A reconstructed network can saved as we did before with :meth:`Qiber3D.Network.save`.
The save function can be given a name for the save file.

.. code-block:: python

    >>> net.save('reconstructed_net.qiber')
    Qiber3D_core [INFO] Network saved to reconstructed_net.qiber

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
    # export the synthetic network as .tif file
    >>> syn_net.export('synthetic.tif', voxel_resolution=voxel_resolution, overwrite=True)
    Qiber3D_render [INFO] Rasterizing network (voxel resolution : 5.00E+00 voxel/unit)
    Qiber3D_core [INFO] Network exported to synthetic.tif.
    # import .tif file
    >>> net = Network.load('synthetic.tif')
    Qiber3D_render [INFO] Rasterizing network (voxel resolution : 5.00E+00 voxel/unit)
    Qiber3D_core [INFO] Network exported to synthetic.tif
    Qiber3D_extract [INFO] Load image data from synthetic.tif
    Qiber3D_extract [INFO] Image voxel size: [0.200,0.200,0.200]
    Qiber3D_extract [INFO] Resample image to cubic voxels
    Qiber3D_extract [INFO] Generate binary representation
    Qiber3D_extract [INFO] Binary representation used a threshold of 29.0% (direct)
    Qiber3D_extract [INFO] reconstruct image
    Qiber3D_reconstruct [INFO] Skeletonize image by thinning
    Qiber3D_reconstruct [INFO] Euclidean distance transformation
    Qiber3D_reconstruct [INFO] Link up skeleton
    Qiber3D_reconstruct [INFO] Build Qiber3D.Network from the raw graph
    Qiber3D_reconstruct [INFO] Cleaning Network
    Qiber3D_reconstruct [INFO] Smooth Segments
    >>> print(net)
    Input file: synthetic.tif
      Number of fibers: 4 (clustered 2)
      Number of segments: 11
      Number of branch points: 4
      Total length: 1120.84
      Total volume: 4665.63
      Average radius: 0.960
      Cylinder radius: 1.151
      Bounding box volume: 799821


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