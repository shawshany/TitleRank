#!/usr/bin/python3
# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2019-12-06 16:39
# @Author   : 病虎
# @E-mail   : victor.xsyang@gmail.com
# @File     : TitleRank.py
# @User     : ora
# @Software: PyCharm
# @Description: 
#Reference:**********************************************
import numpy as np
import pandas as pd
import jieba
from difflib import SequenceMatcher
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix
from wordcloud import WordCloud
from gensim.models import Word2Vec
from scipy import spatial
import sentencepiece as spm
import os

from TitleRank.utils.textprocess import stopwordsdict
from TitleRank.utils.textprocess import remove_stopwords

current_path = os.path.dirname(os.path.abspath(__file__))

class TitleRank():
    def __init__(self, stopwords_path=os.path.join(current_path,'data/stopwords.txt'), userstopwords_path=os.path.join(current_path, 'data/user_stopwords.txt')):
        #导入停用词
        print(stopwords_path)
        if not os.path.isfile(stopwords_path):
            raise Exception("stopwords.txt: file does not exist: " + stopwords_path)
        self.stopwords = set(map(str.strip, open(stopwords_path).readlines()))
        self.title_list=[]
        self.sp = spm.SentencePieceProcessor()
        sp_path = os.path.join(current_path, 'models/news_title_8w.model')
        if not os.path.isfile(sp_path):
            raise Exception("news_title_8w.model: file does not exist: " + sp_path)
        self.sp.load(sp_path)
        sp_model_path = os.path.join(current_path, "embeddings/sp_title.emb")
        self.sp_emb = self.load_models(sp_model_path)

        self.word_vocab_set = set(self.sp_emb.wv.vocab)
        if not os.path.isfile(userstopwords_path):
            raise Exception("userstopwords.txt: file does not exist: " + userstopwords_path)
        self.user_stopwords = stopwordsdict(userstopwords_path)

    def compute_common_chars(self, a, b):
        '''

        :param a: 字符串a
        :param b: 字符串b
        :return: 公共字符个数
        '''
        size = 0
        s = SequenceMatcher(None, a, b)
        blocks = s.get_matching_blocks()
        for block in blocks:
            size += block.size
        return size

    def compute_long_continues_string(self,a,b):
        '''

        :param a: 字符串a
        :param b: 字符串b
        :return: 连续最长公共子串长度
        '''
        s = SequenceMatcher(None,a,b)
        size = s.find_longest_match(0, len(a), 0, len(b)).size
        return size

    def compute_common_words(self,a,b):
        '''

        :param a: 字符串a
        :param b: 字符串b
        :return: 公共word个数
        '''
        #对a、b分别进行分词
        a_words = jieba.lcut(a)
        b_words = jieba.lcut(b)
        intersection = list(set(a_words) & set(b_words))
        return len(intersection)

    def generate_weight(self,a_matrix, title_list,theta=.9):
        '''

        :param a_matrix: 全零矩阵
        :param title_list:文章标题
        :param theta:单词个数阈值（一般默认为0.9 ）
        :return:a_state--状态转移矩阵；a_weight_norm--权重矩阵
        '''
        a_m, a_n = a_matrix.shape
        a_weight = a_matrix.copy()
        a_state = a_matrix.copy()
        for i in range(0, a_m):
            for j in range(0, a_n):
                if i == j:
                    continue
                else:
                    a_weight[i, j] = self.compute_sentence_sim(title_list[i], title_list[j])
                    if (a_weight[i, j] >= theta):
                        a_state[i, j] = 1
                    else:
                        a_weight[i, j] = 0
        a_weight_norm = normalize(a_weight, axis=1, norm='l1')
        return a_state, a_weight_norm

    def pageRank(self, G, G_weight, s=.85, maxerr=.001, steps=1000):
        """
        Computes the pagerank for each of the n states
        Parameters
        ----------
        G: matrix representing state transitions
        Gij is a binary value representing a transition from state i to j.
        G_weight：状态转移权重
        s: probability of following a transition. 1-s probability of teleporting
        to another state.
        maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
        """
        n = G.shape[0]
        # 将 G into 马尔科夫 A
        A = csc_matrix(G, dtype=np.float64)  #
        A_weight = csc_matrix(G_weight, dtype=np.float64)
        rsums = np.array(A.sum(1), dtype=np.float64)[:, 0]  # 沿列求和
        ri, ci = A.nonzero()  # A矩阵中非零数值的横纵坐标
        A.data /= rsums[ri]
        rsums_weight = np.array(A_weight.sum(1), dtype=np.float64)[:, 0]  # 沿列求和
        ri_weight, ci_weight = A_weight.nonzero()  # A_weight矩阵中非零数值的横纵坐标
        A_weight.data /= rsums_weight[ri_weight]
        # A.data = np.multiply(A.data, A_weight.data)
        sink = rsums == 0
        # 计算PR值，直到满足收敛条件
        ro, r = np.zeros(n), np.ones(n)
        count = 0
        while np.sum(np.abs(r - ro)) > maxerr:
            ro = r.copy()
            for i in range(0, n):
                Ai = np.array(A[:, i].todense())[:, 0]
                Ai_weight = np.array(A_weight[:, i].todense())[:, 0]
                # Ai = np.multiply(Ai, Ai_weight)
                Di = sink / float(n)
                Ei = np.ones(n) / float(n)
                r[i] = ro.dot(Ai_weight * s + Di * s + Ei * (1 - s))
                # 归一化
            print("迭代次数{}\n".format(count))
            print(np.sum(np.abs(r - ro)))
            count += 1
            if count > steps:
                break
        return r / float(sum(r))

    def weightPageRank(self, title_list, s=.85, maxerr=.001, steps=1000):
        '''

        :param title_list: 新闻标题
        :param s
        :param maxerr
        :param steps
        :return: 新闻标题pr值及对应的标题，值越大排名越高
        '''
        a = np.zeros((len(title_list), len(title_list)), dtype=np.float64)
        a_state, a_init = self.generate_weight(a, title_list)
        rank_a = self.pageRank(a_state, a_init, s=s, maxerr=maxerr, steps=steps)
        df = pd.DataFrame()
        df['title'] = title_list
        df["score"] = rank_a
        return df

    def word_cloud_fit(self,title_list,max_words=2000):
        '''

        :param title_list: 新闻标题
        :max_words: 词云显示的最大词数
        :return: 词云对象
        '''
        wc = WordCloud(
            max_words=max_words,  # 词云显示的最大词数
            stopwords=self.stopwords,  # 设置停用词
        )
        self.title_list =title_list
        return wc

    def compute_wc_scores(self, df, key, value):
        for i in range(len(df)):
            if key in df.loc[i, "title_split"]:
                df.loc[i, "score"] += value[0]
                df.loc[i, "words"] += ("_" + key + "({}".format(value[0]) + ")")
        return df

    def wordWeightRank(self,wc):
        '''

        :param wc:词云对象
        :param df:
        :return:返回新闻标题及对应的词云权重分数
        '''
        text = ''.join(self.title_list)
        outstr = " ".join(jieba.lcut(text))
        wcloud = wc.generate(outstr)
        word_dict = wcloud.words_
        word_lists = []
        score_lists = []
        for key, value in word_dict.items():
            tmp_key = key.split(" ")
            word_lists.extend(tmp_key)
            score_lists.extend([value] * len(tmp_key))
        #合并为DataFrame格式
        word_fre_frame = pd.DataFrame()
        word_fre_frame["word"] = word_lists
        word_fre_frame["value"] = score_lists
        word_fre_dict = word_fre_frame.set_index('word').T.to_dict('list')
        df = pd.DataFrame()
        df["title"] = self.title_list
        df["score"] = 0.0
        df["words"] = ""
        df["title_split"] = df["title"].apply(lambda x: jieba.lcut(x))
        for key, value in word_fre_dict.items():
            # 计算数值
            df = self.compute_wc_scores(df, key, value)
        return df

    def load_models(self, model_path):
        '''

        :param model_path: sentence_piece分词模型地址
        :return:词向量模型
        '''
        return Word2Vec.load(model_path)

    def avg_feature_vector(self, words_list, model, word_vocab_set, num_features=100):
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in words_list:
            if word in word_vocab_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model.wv[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    def compute_sentence_sim(self, sentence_a, sentence_b):
        '''

        :param sentence_a: 句子a
        :param sentence_b: 句子b
        :return:句子a和b之间的相似度
        '''
        sa_words_list = ",".join(self.sp.EncodeAsPieces(sentence_a)).replace('▁', '').split(',')
        sb_words_list = ",".join(self.sp.EncodeAsPieces(sentence_b)).replace('▁', '').split(',')
        #去除空字符串
        sa_words_list = [item for item in sa_words_list if item != '']
        sb_words_list = [item for item in sb_words_list if item != '']
        sa_words_list = remove_stopwords(sa_words_list,self.user_stopwords)
        sb_words_list = remove_stopwords(sb_words_list,self.user_stopwords)
        s1_afv = self.avg_feature_vector(sa_words_list, self.sp_emb,self.word_vocab_set)
        s2_afv = self.avg_feature_vector(sb_words_list, self.sp_emb,self.word_vocab_set)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        return sim

    def wordvec_sim_rank(self, title_list,theta=0.9):
        '''

        :param title_list: 新闻标题
        :param theta: 相似度阈值，默认0.9
        :return:新闻标题及对应的分数、相似度矩阵
        '''

        matrix = np.zeros((len(title_list), len(title_list)), dtype=np.float64)
        matrix_m, matrix_n = matrix.shape
        sim_weight = matrix.copy()
        sim_state = matrix.copy()
        for i in range(0, matrix_m):
            for j in range(0, matrix_n):
                sim_weight[i, j] = self.compute_sentence_sim(title_list[i], title_list[j])
                if sim_weight[i, j]>=0.9:
                    sim_state[i, j] = 1

        # 计算相似度
        result_lists = []
        for i in range(0, matrix_m):
            result_list=[]
            for j in range(0, matrix_n):
                if sim_state[i,j]==1:
                    result_list.append(j)
            result_lists.append(result_list)

        scores = [len(item_list) for item_list in result_lists]
        df = pd.DataFrame()
        df["title"] = title_list
        df["score"] = scores
        return df, result_lists


    def remove_stopword(self, word):
        '''

        :param word: 指定词汇
        :return: 删除指定词汇
        '''
        self.user_stopwords.pop(word)
        return

    def add_stopword(self,word):
        '''

        :param word: 指定词汇
        :return: 添加指定词汇
        '''
        self.user_stopwords.update(word=word)
        return
