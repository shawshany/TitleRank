#!/usr/bin/python3
# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2019-12-06 16:01
# @Author   : 病虎
# @E-mail   : victor.xsyang@gmail.com
# @File     : setup.py.py
# @User     : ora
# @Software: PyCharm
# @Description: 
#Reference:**********************************************
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TitleRank", # Replace with your own username
    # packages = ['TitleRank'],
    # package_dir={'TitleRank': 'TitleRank'},
    # package_data={'TitleRank': ['data/*','embedding/*','models/*']},
    # include_package_data=True,
    version="0.1.0",
    author="OraYang",
    author_email="victor.xsyang@gmail.com",
    description="A small example package",
    long_description=long_description,
    packages=setuptools.find_packages(),

    long_description_content_type="text/markdown",
    url="https://github.com/shawshany/TitleRank",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)