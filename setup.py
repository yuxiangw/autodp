
from setuptools import setup


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='autodp',
   version='0.1',
   description='Automating Differential Privacy Computation',
   license="Apache",
   long_description=long_description,
   author='Yu-Xiang Wang',
   author_email='yuxiangw@cs.ucsb.edu',
   url='https://github.com/yuxiangw/autodp',
   packages=['autodp'],  #same as name
   install_requires=['numpy', 'scipy'], #external packages as dependencies
)