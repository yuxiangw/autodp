
from setuptools import setup


with open("README.md", 'r') as f:
    long_description = f.read()

def _parse_requirements(path):
  """Parses requirements from file."""
  with open(os.path.join(here, path)) as f:
    deps = []
    for line in f:
      deps.append(line.rstrip())
    return deps

setup(
   name='autodp',
   version='0.1.1',
   description='Automating Differential Privacy Computation',
   license="Apache",
   long_description="The package helps researchers and developers to correctly use advanced methods in differential privacy and obtain provable DP guarantees. The core of the package is an analytical moments accountant that keeps track of Renyi Differential Privacy in analytical forms.",
   author='Yu-Xiang Wang',
   author_email='yuxiangw@cs.ucsb.edu',
   url='https://github.com/yuxiangw/autodp',
   download_url = 'https://github.com/yuxiangw/autodp/archive/v0.1.1.tar.gz',
   keywords = ['Differential Privacy','Moments Accountant','Renyi Differential Privacy'],
   packages=['autodp'],  #same as name
   install_requires=[_parse_requirements('requirements.txt')], #external packages as dependencies
   classifiers=['Development Status :: 3 - Alpha',
   'Intended Audience :: Developers',
   'Intended Audience :: Science/Research',
   'Topic :: Scientific/Engineering :: Artificial Intelligence',
   'Topic :: Security',
   'License :: OSI Approved :: Apache Software License',
   'Programming Language :: Python :: 3',
   ],
)
