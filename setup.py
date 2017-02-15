# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='template',
    version='0.0.1',
    description='Template project',
    long_description=readme,
    author='Michal Koperski',
    author_email='michal.koperski@inria.fr',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

