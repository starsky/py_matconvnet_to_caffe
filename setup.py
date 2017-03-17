# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='py_matconvnet_2_caffe',
    version='0.0.1',
    description='Py MatConvNet2Caffe',
    long_description=readme,
    author='Michal Koperski',
    author_email='michal.koperski@inria.fr',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    package_data={'py_matconv_to_caffe.matlab_scripts': ['*.m']},
    # include_package_data=True,
    scripts=['py_matconv_to_caffe/matlab_scripts/py_matconv2caffe_prepare_test_data'],
    entry_points={'console_scripts': ['py_matconv2caffe_convert=py_matconv_to_caffe.matconv_to_caffe.py:main']
                  }
)

