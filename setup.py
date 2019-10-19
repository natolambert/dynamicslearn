from distutils.core import setup
from setuptools import find_packages

setup(
    name='dynamicslearn',
    version='0.1dev',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=[
          'hydra-core==0.9.0',
      ],
)  
