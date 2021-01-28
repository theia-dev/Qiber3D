


Visualization
-------------

To explore some of the possibilities of the :class:`Qiber3D.Render` module we can use the synthetic network as example.

.. code-block:: python

    >>> from Qiber3D import IO
    >>> net = IO.load.synthetic_network()
    >>> print(net)
    Input file: memory
      Number of fibers: 4 (clustered 2)
      Number of segments: 11
      Number of branch points: 5
      Total length: 1141.44
      Total volume: 4688.67
      Average radius: 0.936
      Cylinder radius: 1.143
      Bounding box volume: 806162

A loaded network can be visualized in different ways.
Calling `render.show()` gives a quick view of the network.

.. code-block:: python

    net.render.show()

.. only:: not html

   .. image:: /img/synthetic_show.*
      :width: 50%
      :align: center

.. x3d:: "_static/x3d/synthetic_show.x3d"

The ``color_mode`` parameter changes how the different segments (:class:`Qiber3D.Segment`) are represented.
Selecting ``'fiber'`` randomly colors fibers (:class:`Qiber3D.Fiber`) that have at least one branch point.
Fibers without a branch point are grey.
With ``'segment'`` all segments are colored randomly.
The full list of possible ``color_mode`` parameter is documented with :meth:`Qiber3D.Render.show`.

.. code-block:: python

    net.render.show(color_mode='fiber')

.. only:: not html

   .. image:: /img/synthetic_show_fiber.*
      :width: 50%
      :align: center

.. x3d:: "_static/x3d/synthetic_show_fiber.x3d"

.. code-block:: python

    net.render.show(color_mode='segment')

.. only:: not html

   .. image:: /img/synthetic_show_fiber.*
      :width: 50%
      :align: center

.. x3d:: "_static/x3d/synthetic_show_segment.x3d"

Sometimes it can be helpful to display just the reconstructed center-lines of a network.
To archive this the parameter ``object_type`` can be set tp ``'line'``.

.. code-block:: python

    net.render.show(color_mode='flat', color=(0, 0, 0), object_type='line')

.. only:: not html

   .. image:: /img/synthetic_show_segments_line_flat.*
      :width: 50%
      :align: center

.. x3d:: "_static/x3d/synthetic_show_segment_line_flat.x3d"

While interactive representations are helpful when inspecting a small number of networks, it is more effective to create different views of the network as rendered images.
For this purpose :meth:`Qiber3D.Render.overview` can be used.
The syntax is very similar to :meth:`Qiber3D.Render.show`, but now a ``out_path`` and the ``image_resolution`` can be set.
If no ``out_path`` is set the file name is automatically chosen.
An existing file will not be overwritten if not ``overwrite`` is set to ```True``.

.. code-block:: python

   >>> net.render.overview(color_mode='segment_length', color_map='magma', background='red')
   Qiber3D_render [INFO] New overview saved under: overview_segment_length_synthetic.png
   >>> net.render.overview(color_mode='segment_length', color_map='magma', background='red')
   Qiber3D_helper [WARNING] File exist: overview_segment_length_synthetic.png


.. image:: /img/overview_segment_length_synthetic.*
      :width: 50%
      :align: center

While we requested a red background, it is not visible in the resulting image.
The reason for this behavior is, that for the background the alpha channel of the :file:`.png` file comes into play.
If a image without transperency is needed ``rgba`` can be set to ```False``.

.. code-block:: python

   >>> net.render.overview(color_mode='segment_length', color_map='magma', background='red', rgba=False, overwrite=True)
   Qiber3D_render [INFO] New overview saved under: overview_segment_length_synthetic.png

.. image:: /img/overview_segment_length_synthetic_RGB.*
      :width: 50%
      :align: center

The last basic visualization option is to save an animation of the network as a :file:`.mp4` movie.

.. code-block:: python

    >>> net.render.animation(color_mode='segment', color_map='hsv', duration=4, background=(1.0, 1.0, 1.0))
    Qiber3D_render [INFO] Preparing animation
    rendering: 100%|███████████████████████████████████████████████████████████████████| 120/120 [00:06<00:00, 17.94frame/s]
    Qiber3D_render [INFO] New animation saved under: animation_segment_synthetic.mp4

.. only:: not html

   .. image:: /img/synthetic_show_segments_line_flat.*
      :width: 50%
      :align: center

.. video-loop:: animation_segment_synthetic




