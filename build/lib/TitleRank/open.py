#!/usr/bin/python3
# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2019-12-10 19:59
# @Author   : 病虎
# @E-mail   : victor.xsyang@gmail.com
# @File     : open.py
# @User     : ora
# @Software: PyCharm
# @Description: 
#Reference:**********************************************
stopwords_path="TitleRank/data/stopwords.txt"
def load_stopwords(stopwords_path="TitleRank/data/stopwords.txt"):
    return set(map(str.strip, open(stopwords_path).readlines()))
STOPWORDS = load_stopwords()
