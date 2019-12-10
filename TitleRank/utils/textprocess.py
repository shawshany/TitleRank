#!/usr/bin/python3
# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2019-12-09 16:23
# @Author   : 病虎
# @E-mail   : victor.xsyang@gmail.com
# @File     : textprocess.py
# @User     : ora
# @Software: PyCharm
# @Description: 
#Reference:**********************************************
def stopwordsdict(filepath='TitleRank/data/user_stopwords.txt'):
    '''

    :param filepath: 用户指定停用词路径
    :return: 停用词词典
    '''
    stopwords = dict()
    for line in open(filepath, 'r', encoding='utf-8').readlines():
        stopwords[line.strip()] = True
    return stopwords

def remove_stopwords(words_list,userstopwords):
    '''
    :param words_list: 分词列表
    :param userstopwords: 用户指定停用词表
    :return: 去除停用词后的列表
    '''
    data = []
    for word in words_list:
        if word not in userstopwords.keys():
            if word != '\t':
                data.append(word)
    return data