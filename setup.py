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
      author=config.app_author,
      author_email='',
      license='MIT',
      packages=['Qiber3D'],
      include_package_data=True,
      install_requires=["numpy", "jinja2", "matplotlib", "tqdm", "scikit-image", "scipy", "pims", "vedo",
                        "networkx", "nd2reader", "blosc", "openpyxl", "lxml"],
      extras_require={
          'kimimaro': ['kimimaro']
      },
      zip_safe=True,
      keywords=['Skeleton', 'Network', 'Fiber', 'Reconstruction', 'Neurons',
                'Vessel', 'Vascular', 'confocal', 'microscopy'],
      python_requires='~=3.7',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',

          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Information Analysis',

          'License :: OSI Approved :: MIT License',

          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3 :: Only'
      ],
      )
