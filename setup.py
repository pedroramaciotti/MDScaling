"""`setup.py`"""
from setuptools import setup, find_packages

# # Package requirements
# with open('requirements.txt') as f:
#     INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

INSTALL_REQUIRES=[
'numpy',
'pandas',
'scikit-learn',
'scipy',
'matplotlib',
'prince',
]

setup(name='mdscaling',
      version='0.0.6',
      description='Multi-dimensional network scaling',
      author='Pedro Ramaciotti Morales',
      author_email='pedro.ramaciotti@gmail.com',
      url = 'https://github.com/pedroramaciotti/MDScaling',
      download_url = 'https://github.com/pedroramaciotti/MDScaling/archive/0.0.6.tar.gz',
      keywords = ['multi-dimensional scaling','graph compression','dimentionality reduction','representation learning'],
      packages=find_packages(),
      data_files=[('', ['LICENSE'])],
      install_requires=INSTALL_REQUIRES)
