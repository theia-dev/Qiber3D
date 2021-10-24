# Qiber3D <img src='https://github.com/theia-dev/Qiber3D/raw/master/docs/img/synthetic_animation_silver.gif' align="right"/>
Automated quantification of fibrous networks

[![PyPi](https://img.shields.io/pypi/v/qiber3d.svg?style=for-the-badge)](https://pypi.org/project/Qiber3D/)
[![Status](https://img.shields.io/pypi/status/qiber3d.svg?style=for-the-badge)](https://pypi.org/project/Qiber3D/)

[![Documentation](https://img.shields.io/readthedocs/qiber3d.svg?style=for-the-badge)](https://Qiber3D.readthedocs.io)

[![License](https://img.shields.io/github/license/theia-dev/qiber3d.svg?style=for-the-badge)](https://github.com/theia-dev/Qiber3D/blob/master/LICENSE.txt)
[![Github issues](https://img.shields.io/github/issues/theia-dev/qiber3d.svg?style=for-the-badge)](https://github.com/theia-dev/Qiber3D/issues)

[![Coverage](https://img.shields.io/codecov/c/gh/theia-dev/Qiber3D?token=UCNHVP172J&style=for-the-badge)](https://app.codecov.io/gh/theia-dev/Qiber3D)
[![Build](https://img.shields.io/github/workflow/status/theia-dev/Qiber3D/Qiber3D.svg?style=for-the-badge)](https://github.com/theia-dev/Qiber3D/actions/workflows/test-Qiber3D.yml)

[![Binder](https://img.shields.io/badge/BINDER-Latest-brightgreen?style=for-the-badge)](https://mybinder.org/v2/gh/theia-dev/Qiber3D_jupyter/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Ftheia-dev%252FQiber3D%26urlpath%3Dtree%252FQiber3D%252Fdocs%252Fjupyter%252Findex.ipynb)
[![Binder](https://img.shields.io/badge/BINDER-Dev-blue?style=for-the-badge)](https://mybinder.org/v2/gh/theia-dev/Qiber3D_jupyter/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Ftheia-dev%252FQiber3D%26urlpath%3Dtree%252FQiber3D%252Fdocs%252Fjupyter%252Findex.ipynb%26branch%3Ddev)

## Setup
    pip install Qiber3D
    
You can install the _latest_ version

    pip install -U git+https://github.com/theia-dev/Qiber3D.git#egg=Qiber3D

or the _dev_ version directly from GitHub.
    
    pip install -U git+https://github.com/theia-dev/Qiber3D.git@dev#egg=Qiber3D

    
## Quick usage

An image stack or a preprocessed network can be loaded with ``Network.load()``
To follow this example, you can download the image stack from figshare under [doi:10.6084/m9.figshare.13655606](https://doi.org/10.6084/m9.figshare.13655606) or use the `Example` class.
```python
import logging
from Qiber3D import Network, config
from Qiber3D.helper import Example, change_log_level

config.extract.nd2_channel_name = 'FITC'
change_log_level(logging.DEBUG)

net_ex = Example.nd2()
net = Network.load(net_ex)
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

A more extensive interactive example is available as a Jupyter notebook.
You can try it out directly on [Binder](https://mybinder.org/v2/gh/theia-dev/Qiber3D_jupyter/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Ftheia-dev%252FQiber3D%26urlpath%3Dtree%252FQiber3D%252Fdocs%252Fjupyter%252Findex.ipynb).
More in-depth documentation, including details on the inner working, can be found at [Read the docs](https://Qiber3D.readthedocs.io).


The complete source code is available on [GitHub](https://github.com/theia-dev/Qiber3D).
