# Qiber3D <img src='https://github.com/theia-dev/Qiber3D/raw/master/docs/img/synthetic_animation_silver.gif' align="right"/>
Automated quantification of fibrous networks

[![PyPi](https://img.shields.io/pypi/v/qiber3d.svg?style=for-the-badge)](https://pypi.org/project/Qiber3D/)
[![Status](https://img.shields.io/pypi/status/qiber3d.svg?style=for-the-badge)](https://pypi.org/project/Qiber3D/)

[![Documentation](https://img.shields.io/readthedocs/qiber3d.svg?style=for-the-badge)](https://Qiber3D.readthedocs.io)

[![License](https://img.shields.io/github/license/theia-dev/qiber3d.svg?style=for-the-badge)](https://github.com/theia-dev/Qiber3D/blob/master/LICENSE.txt)
[![Github issues](https://img.shields.io/github/issues/theia-dev/qiber3d.svg?style=for-the-badge)](https://github.com/theia-dev/Qiber3D/issues)

[![Coverage](https://img.shields.io/codecov/c/gh/theia-dev/Qiber3D?token=UCNHVP172J&style=for-the-badge)](https://app.codecov.io/gh/theia-dev/Qiber3D)
[![Build](https://img.shields.io/github/workflow/status/theia-dev/Qiber3D/Qiber3D.svg?style=for-the-badge)](https://github.com/theia-dev/Qiber3D/actions/workflows/test-Qiber3D.yml)


## Setup
    pip install Qiber3D
    
You can also install the latest version directly from GitHub.

    pip install -U git+https://github.com/theia-dev/Qiber3D.git#egg=Qiber3D

    
## Usage

An image stack, or a preprocessed network can be loaded with ``Network.load()``
To follow this example you can download the image stack from figshare under [doi:10.6084/m9.figshare.13655606](https://doi.org/10.6084/m9.figshare.13655606).

```python
import logging
from Qiber3D import Network, config

config.extract.nd2_channel_name = 'FITC'
config.log_level = logging.DEBUG

net = Network.load('microvascular_network.nd2')
print(net)
# Input file: microvascular_network.nd2
#   Number of fibers: 459 (clustered 97)
#   Number of segments: 660
#   Number of branch points: 130
#   Total length: 16056.46
#   Total volume: 1240236.70
#   Average radius: 4.990
#   Cylinder radius: 4.959
#   Bounding box volume: 681182790

net.save(save_steps=True)
# Qiber3D_core [INFO] Network saved to Exp190309_PrMECs-NPF180_gel4_ROI-c.qiber

net.render.show()
net.render.compare()
```

A quick way to explore the possibilities of ``Qiber3D`` is the use of the synthetic network.
```python
from Qiber3D import IO
net = IO.load.synthetic_network()
print(net)
# Input file: memory
#   Number of fibers: 4 (clustered 2)
#   Number of segments: 11
#   Number of branch points: 5
#   Total length: 1141.44
#   Total volume: 4688.67
#   Average radius: 0.936
#   Cylinder radius: 1.143
#   Bounding box volume: 806162
net.length
# 1141.437678088988
net.volume
# 4688.667104530579
net.fiber[0]
# Fiber 0 l=451.65, V=1651.71
print(net.fiber[0])
# Fiber ID: 0
#   Number of segments: 5
#   Total length: 451.65
#   Total volume: 1651.71
#   Average radius: 0.86
#   Cylinder radius: 1.08
net.fiber[0].segment
# {0: Segment 0 l=40.19, V=17.30, 1: Segment 1 l=124.71, V=113.51, 2: Segment 2 l=32.72, V=112.79, 
#  3: Segment 3 l=154.15, V=1080.55, 4: Segment 4 l=99.88, V=327.56}
print(net.segment[1])
# Segment ID: 1
#   Number of parts: 200
#   Total length: 124.71
#   Total volume: 113.51
#   Average radius: 0.51
#   Cylinder radius: 0.54
net.render.show()
net.render.show(color_mode='segment', color_map='hsv')
net.figure.directions(out_path=None)
```

The full source code can be accessed on [GitHub](https://github.com/theia-dev/Qiber3D) with the corresponding documentation hosted at [Read the docs](https://Qiber3D.readthedocs.io).
