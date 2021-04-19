# -*- coding=utf-8 -*-
# @Time :2021/4/12 20:40
# @Author :Hobbey
# @Site : 
# @File : crftest.py
# @Software : PyCharm
from test import *
import sys
import re
from collections import defaultdict

def get_feature_functions(word_sets, labels, observes):
    """生成各种特征函数"""
    print("get feature functions ...")
    transition_functions = [
        lambda yp, y, x_v, i, _yp=_yp, _y=_y: 1 if yp == _yp and y == _y else 0
        for _yp in labels[:-1] for _y in labels[1:]
        ]
    def set_membership(tag, word_sets):
        def fun(yp, y, x_v, i):
            if i < len(x_v) and x_v[i].lower() in word_sets[tag]:
                return 1
            else:
                return 0
        return fun

    observation_functions = [set_membership(t, word_sets) for t in word_sets]

    misc_functions = [
        lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[^0-9a-zA-Z]+$', x_v[i]) else 0,
        lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[A-Z\.]+$', x_v[i]) else 0,
        lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[0-9\.]+$', x_v[i]) else 0
    ]

    tagval_functions = [
        lambda yp, y, x_v, i, _y=_y, _x=_x: 1 if i < len(x_v) and y == _y and x_v[i].lower() == _x else 0
        for _y in labels
        for _x in observes]

    return transition_functions + tagval_functions + observation_functions + misc_functions

if __name__ == '__main__':
    word_data = []
    label_data = []
    all_labels = set()
    word_sets = defaultdict(set) ## 默认值为空的set列表
    observes = set()
    lineNumber=0
    for line in open("sample_2.txt",encoding='utf-8'):
        # print(line)
        words, labels = [], []
        lineNumber+=1
        for token in line.strip().split():
            word, label = token.split('/')
            all_labels.add(label)
            word_sets[label].add(word.lower())
            observes.add(word.lower())
            words.append(word)
            labels.append(label)
        word_data.append(words)
        label_data.append(labels)
        if lineNumber>6:
            break
    labels = ['START']+ list(all_labels)+ ['END']
    feature_functions = get_feature_functions(word_sets, labels, observes)

    crf = CRF(labels=labels, feature_functions=feature_functions)
    crf.train(word_data, label_data)
    s = 0
    tol_len=0
    for x_vec, y_vec in zip(word_data[-5:], label_data[-5:]):
        # print("raw data: ", x_vec)
        print("prediction: ", pred:=crf.predict(x_vec))
        print("truth label: ", y_vec)
        pre_res = pred
        y_vec_ = y_vec[1:-1]
        length = len(y_vec_)
        tol_len+=length
        for i in range(length):
            s += (pre_res[i] == y_vec_[i])
    print(s/tol_len)


