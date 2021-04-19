# -*- coding=utf-8 -*-
# @Time :2021/4/5 16:00
# @Author :Hobbey
# @Site : 
# @File : test.py
# @Software : PyCharm
# 条件随机场
# 实现CRF的训练和使用，语料的预处理，标注结果的评估
import numpy as np
from scipy import special
from scipy import optimize

class CRF(object):
    def __init__(self, feature_functions, labels):
        """
        :param feature_functions: 输入的特征函数
        :param labels: 输入的训练数据标注
        """
        # 特征函数
        self.ft_func = feature_functions
        # 特征函数的权值
        self.w = np.random.rand(len(self.ft_func))  # 一个特征标签对应一个特征方程
        # labels
        self.labels = labels
        print(labels)
        self.START = labels[0]  ## 标签数量要加二
        self.END = labels[-1]
        # label_id 的字典
        self.label_id = {value: idx for idx, value in enumerate(self.labels)}

    def get_all_features(self, x_vec):
        """
        :param x_vec:输入的观测序列
        :return: x_vec序列中所有（y'，y）组成的特征值
        result size:[len(x_vec)+1, Y, Y, K]
        """
        result = np.zeros((len(x_vec) + 1, len(self.labels), len(self.labels), len(self.ft_func)),dtype=np.int8)
        for i in range(len(x_vec) + 1):
            for j, yp in enumerate(self.labels):
                for k, y in enumerate(self.labels):
                    for l, f in enumerate(self.ft_func):
                        result[i, j, k, l] = f(yp, y, x_vec, i)  ## 0 / 1
        return result

    def log_dot_mv(self, logM, logB):
        """
        矩阵乘向量
        :param logM: 矩阵，本项目中特征矩阵的取对数
        :param logB: 向量，本项目中后向向量取对数
        """
        # np.expand_dims(x, axis=0) 转换为行向量
        # np.sum(x, axis=1) 对矩阵的每一行相加
        # logsumexp(x, axis=0/1) = np.log(np.sum(np.exp(x), axis=0/1))
        return special.logsumexp(logM + np.expand_dims(logB, axis=0), axis=1)

    def log_dot_vm(self, logA, logM):
        """
    	向量乘矩阵
        :param loga:
        :param logM:
        :return:
        """
        return special.logsumexp(np.expand_dims(logA, axis=1) + logM, axis=0)

    def forward(self, log_M_s, start):
        """

        :param log_M_s: 特征函数矩阵取对数
        :param start: 在状态序列添加的start
        :return: 前向向量矩阵

        初始化 alpha_0 = 1, if y = start ;else 0
        取对数之后为：alpha_0 = log 1 = 0 if y=start else log 0 = -inf

        alpha size: 因为添加了一维start，假设特征矩阵维度(m,m),那么alpha维度（m+1,m）
        """
        T = log_M_s.shape[0]  ##  序列长度
        Y = log_M_s.shape[1]  ##  状态个数

        alphas = np.NINF * np.ones((T + 1, Y)) ## 较小的数
        alpha = alphas[0]
        # log 1 = 0
        alpha[start] = 0
        for t in range(1, T + 1):
            alphas[t] = self.log_dot_vm(alpha, log_M_s[t - 1])
            alpha = alphas[t]
        return alphas

    def backfarward(self, log_M_s, end):
        """

        :param log_M_s:
        :param logb:
        :return:
        """
        T = log_M_s.shape[0]
        Y = log_M_s.shape[1]

        betas = np.NINF * np.ones(shape=(T + 1, Y))
        ## 顺序一致方便后续使用查找
        beta = betas[-1]
        beta[end] = 0
        for t in range(T - 1, -1, -1):
            betas[t] = self.log_dot_mv(log_M_s[t], beta)
            beta = betas[t]
        return betas

    def creat_vector_list(self, x_vecs, y_vecs):
        """

        :param x_vecs:输入的x（词语）
        :param y_vecs:输入的y（词性labels）
        :return: 观测序列和labels
        """
        print('create observation and label list')
        print('total training data :', len(x_vecs))
        ## 数据处理
        observations = [self.get_all_features(x_vec) for x_vec in x_vecs]
        labels = [None] * len(y_vecs)
        for i in range(len(y_vecs)):
            assert (len(y_vecs[i]) == len(x_vecs[i]))
            # 加入start和end，借助CRF的矩阵形式建模
            y_vecs[i].insert(0, self.START)
            y_vecs[i].append(self.END)
            labels[i] = np.array([self.label_id[y] for y in y_vecs[i]], copy=False, dtype=np.int8)
        return observations, labels

    def train(self, x_vecs, y_vecs):
        """

        :param x_vecs:
        :param y_vecs:
        :param debug:
        :return:
        """
        vectorized_x_vecs, vectorized_y_vecs = self.creat_vector_list(x_vecs, y_vecs)
        print('start training')
        l = lambda w: self.neg_likelihood_and_deriv(vectorized_x_vecs, vectorized_y_vecs, w)
        #
        val = optimize.fmin_l_bfgs_b(l, self.w)  # 返回三元组，loss最小的参数，最小loss值，信息字典蕴含比如梯度，迭代次数等

        self.w, _, _ = val
        return self.w  ## 更新了一次w的值,因为每句话长度不同只能一次次更新

    def neg_likelihood_and_deriv(self, x_vec_list, y_vec_list, w, debug=False):
        """
        求负的对数似然函数，以及似然函数对权重w的导数
        :param x_vecs:
        :param y_vecs:
        :param w:
        :param debug:
        :return:
        """
        # 初始化
        likelihood = 0
        derivative = np.zeros(len(self.w))

        # 对观测序列X中的每一个位置
        for x_vec, y_vec in zip(x_vec_list, y_vec_list):
            all_features = x_vec
            length = x_vec.shape[0]
            # 下边代码中的y_vec = START + y_vec + END
            yp_vec_ids = y_vec[:-1]
            y_vec_ids = y_vec[1:]
            # log_M_s: len(x_vec)+1, Y, Y
            log_M_s = np.dot(all_features, w)
            # alphas: len(x_vec)+2, Y
            log_alphas = self.forward(log_M_s, self.label_id[self.START])
            last = log_alphas[-1]
            # betas: len(x_vec)+2, Y
            log_betas = self.backfarward(log_M_s, self.label_id[self.END])
            # Z = alpha[-1](x)*1 = 1*beta[0]
            log_Z = special.logsumexp(last)
            # reshape alphas的个子序列变为列向量，beta的，每一个子序列变为行向量
            log_alphas1 = np.expand_dims(log_alphas[1:], axis=2)
            log_betas1 = np.expand_dims(log_betas[:-1], axis=1)
            # log_probs : len(x_vec)+1, Y, Y
            log_probs = log_alphas1 + log_M_s + log_betas1 - log_Z
            log_probs = np.expand_dims(log_probs, axis=3)
            # 计算特征函数关于模型的期望，也就是关于条件概率P(Y|X)的期望
            # ,axis=(0,1,2)表示在所有维度上都相加，最后得到一个数
            exp_features = np.sum(np.exp(log_probs) * all_features, axis=(0, 1, 2))
            # 计算特征函数关于训练数据的期望,也就是关于联合分布P(X,Y)的期望
            emp_features = np.sum(all_features[range(length), yp_vec_ids, y_vec_ids], axis=0)
            # 计算似然函数
            likelihood += np.sum(log_M_s[range(length), yp_vec_ids, y_vec_ids]) - log_Z
            # 计算偏导数
            derivative += emp_features - exp_features
        return -likelihood, -derivative #-最小化=最大化

    def predict(self, x_vec):
        """
        Viterbi算法预测
        :param x_vec:
        :param debug:
        :return:
        """
        # all_features: T + 1, Y, Y, K
        all_features = self.get_all_features(x_vec)
        # 非规范化概率 = w*feature
        log_potential = np.dot(all_features, self.w)
        T = len(x_vec)
        Y = len(self.labels)
        # Psi 保存每个时刻最优路径的下标
        Psi = np.ones((T, Y), dtype=np.int8) * -1
        # viterbi算法的初始化
        delta = log_potential[0, 0]
        # 递推
        for t in range(1, T):
            next_delta = np.zeros(Y)
            for y in range(Y):
                # t:第t时刻的子序列；y：第yi个label
                # log_potential[t, :, y]表示第t时刻第y个label的权重w
                w = delta + log_potential[t, :, y]
                Psi[t, y] = psi = w.argmax()
                next_delta[y] = w[psi]
            delta = next_delta
        # 回溯最优路径
        y = delta.argmax()
        trace = []
        for t in range(T - 1, -1, -1):
            trace.append(y)
            y = Psi[t, y]
        trace.reverse()
        return [self.labels[i] for i in trace] ## 还原词性标注序列
