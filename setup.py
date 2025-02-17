__author__ = 'Brian M Anderson'
# Created on 9/15/2020


from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ImageProcessorsModule',
    author='Brian Mark Anderson',
    author_email='markba122@gmail.com',
    version='0.0.2',
    description='Services for processing and creating tensorflow or pytorch records',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'ImageProcessorsModule': 'src/Processors'},
    packages=['ImageProcessorsModule'],
    include_package_data=True,
    url='https://github.com/brianmanderson/https://github.com/brianmanderson/Image_Processors_Module',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    install_requires=required,
)
