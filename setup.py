import sys
from pathlib import Path

from setuptools import setup

base_dir = Path(__file__).absolute().parent
sys.path.insert(0, str(base_dir / 'Qiber3D'))

import config  # import Qiber3D config without triggering the module __init__.py

read_me = base_dir / 'README.md'
long_description = read_me.read_text(encoding='utf-8')
version = config.version

setup(name=config.app_name,
      version=version,
      description='Automated quantification of fibrous networks',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url=config.url,
      download_url=f'https://github.com/theia-dev/Qiber3D/archive/v{version}.zip',
      project_urls={"Binder": "https://mybinder.org/v2/gh/theia-dev/Qiber3D_jupyter/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Ftheia-dev%252FQiber3D%26urlpath%3Dtree%252FQiber3D%252Fdocs%252Fjupyter%252Findex.ipynb",
                    "Documentation": "https://qiber3d.readthedocs.io/"
                    },
      author=config.app_author,
      author_email='',
      license='MIT',
      packages=['Qiber3D'],
      include_package_data=True,
      install_requires=["nd2reader>=3.3.0", "vedo>=2023.5.0", "PIMS>=0.6.1", "tifffile>=2023.9.26", "networkx>=3.2.1",
                        "matplotlib>=3.8.2", "blosc>=1.11.1", "openpyxl>=3.1.2", "vtk==9.3.0", "kimimaro>=3.0.0",
                        "scipy>=1.11.3", "tqdm>=4.66.1", "scikit-image>=0.22.0", "numpy>=1.26.2", "Pillow>=10.1.0"],
      extras_require={
          'kimimaro': ['kimimaro']
      },
      zip_safe=True,
      keywords=['Skeleton', 'Network', 'Fiber', 'Reconstruction', 'Neurons',
                'Vessel', 'Vascular', 'confocal', 'microscopy'],
      python_requires='~=3.7',
      classifiers=[
          'Development Status :: 4 - Beta',

          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Information Analysis',

          'License :: OSI Approved :: MIT License',

          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3 :: Only'
      ],
      )
