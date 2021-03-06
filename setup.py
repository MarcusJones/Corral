#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

# requirements = ['keeper-contracts']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="test",
    author_email='test@testing222.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Corral - Training for RC autonomous driving",
    license="Apache Software License 2.0",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='corral',
    name='corral',
    packages=find_packages(include=['corral']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MarcusJones/corral',
    version='0.0.1',
    zip_safe=False,
)
