# -*- coding=utf-8 -*-
# @Time :2021/4/21 19:22
# @Author :Hobbey
# @Site : 
# @File : SVM.py
# @Software : PyCharm
import numpy as np
import random


class SVM(object):
    def __init__(self, X, y_label, C, max_iter, kernel=('rbf',1.3)):
        '''
        :param X: data
        :param y_label: data_label
        :param C: penalty factor
        :param max_iter: the max number of iteration
        :param kernel: kernel function
        '''
        self.data = X
        self.label = y_label
        self.C = C
        self.max_iter = max_iter
        self.kernel = kernel
        self.x_num=len(self.data)
        self.alpha=np.random.random(size=self.x_num)
        self.beta=random.random()

        self.K=np.zeros(shape=(self.x_num,self.x_num))
        if self.kernel[0]=='rbf':# 高斯核
            self.ker_function=lambda x:np.exp(-np.sum((self.data-x)**2,axis=1)/self.kernel[1])
        elif self.kernel[0]=='linear':# 线性核
            self.ker_function=lambda x:np.dot(self.data,x)
        elif self.kernel[0]=='sigmoid':# S核
            self.ker_function=lambda x:np.tanh(np.dot(self.data,x))
        elif self.kernel[0] =='multiply': # 多项式核
            self.ker_function=lambda x:np.power(np.dot(self.data,x)+1,kernel[1])
        for i in range(self.x_num):
            self.K[i,:]=self.ker_function(self.data[i,:])
        self.E=(np.dot(self.alpha*self.label,self.K)+self.beta-self.label).ravel()
    def trian(self):
        iter=0
        alpha_changed=True
        KKT_check=True
        while iter<self.max_iter and (alpha_changed or KKT_check):
            alpha_changed=False
            if KKT_check:
                for i in range(self.x_num):
                    if (self.E[i]*self.label[i]<-0.001) and (self.alpha[i]<self.C) or (self.E[i]*self.label[i]>0.001 and self.alpha[i]>self.C):
                        print('1')
                        max_deta = 0
                        k = 0
                        for j in range(self.x_num):
                            if j == i:
                                continue
                            deltaE = abs(self.E[i] - self.E[j])
                            if deltaE > max_deta:
                                max_deta = deltaE
                                k = j
                        if self.label[i] != self.label[k]:
                            L = max(0, self.alpha[k] - self.alpha[i])
                            H = min(self.C, self.C + self.alpha[k] - self.alpha[i])
                        else:
                            L = max(0, self.alpha[k] + self.alpha[i] - self.C)
                            H = min(self.C, self.alpha[k] + self.alpha[i])
                        eta = self.K[i, i] + self.K[k, k] - 2 * self.K[i, k]
                        if  eta>0 and L != H:
                            alpha_changed = True
                            alpha_kold = self.alpha[k]
                            alpha_iold = self.alpha[i]
                            self.alpha[k] += self.label[k] * (self.E[i] - self.E[k]) / eta
                            self.alpha[k] = self.clipalpha(self.alpha[k], L, H)
                            self.alpha[i] += self.label[i] * self.label[k]*(alpha_kold - self.alpha[k])

                            b1 = self.beta - self.E[i] - self.label[i] * (self.alpha[i] - alpha_iold) * self.K[i, i] - \
                                 self.label[k] * (
                                         self.alpha[k] - alpha_kold) * self.K[i, k]
                            b2 = self.beta - self.E[k] - self.label[i] * (self.alpha[i] - alpha_iold) * self.K[i, k] - \
                                 self.label[k] * (
                                         self.alpha[k] - alpha_kold) * self.K[k, k]
                            if 0 < self.alpha[i] < self.C:
                                self.b = b1
                            elif 0 < self.alpha[k] < self.C:
                                self.b = b2
                            else:
                                self.b = (b1 + b2) / 2.0
                            self.E = (np.dot(self.alpha * self.label, self.K) + self.beta-self.label).ravel()
                            print('{} is starting'.format(i))
                            if abs(self.alpha[k] - alpha_kold) < 0.000001:
                                print('{} not moving enough iter {}'.format(k,i))
            else:
                for i in range(self.x_num):
                    if 0<self.alpha[i]<self.C:
                        if (self.E[i] * self.label[i] < -0.001 and self.alpha[i] < self.C) or (
                                self.E[i] * self.label[i] > 0.001 and self.alpha[i] > self.C):
                            max_deta = 0
                            k = 0
                            for j in range(self.x_num):
                                if j == i:
                                    continue
                                deltaE = abs(self.E[i] - self.E[j])
                                if deltaE > max_deta:
                                    max_deta = deltaE
                                    k = j
                            if self.label[i] != self.label[k]:
                                L = max(0, self.alpha[k] - self.alpha[i])
                                H = min(self.C, self.C + self.alpha[k] - self.alpha[i])
                            else:
                                L = max(0, self.alpha[k] + self.alpha[i] - self.C)
                                H = min(self.C, self.alpha[k] + self.alpha[i])
                            eta = self.K[i, i] + self.K[k, k] - 2 * self.K[i, k]
                            if eta > 0 and L != H:
                                alpha_changed = True
                                print('2')
                                alpha_kold = self.alpha[k]
                                alpha_iold = self.alpha[i]
                                self.alpha[k] += self.label[k] * (self.E[i] - self.E[k]) / eta
                                self.alpha[k] = self.clipalpha(self.alpha[k], L, H)
                                self.alpha[i] += self.label[i] * self.label[k] * (alpha_kold - self.alpha[k])

                                b1 = self.beta - self.E[i] - self.label[i] * (self.alpha[i] - alpha_iold) * self.K[i, i] - \
                                     self.label[k] * (
                                             self.alpha[k] - alpha_kold) * self.K[i, k]
                                b2 = self.beta - self.E[k] - self.label[i] * (self.alpha[i] - alpha_iold) * self.K[i, k] - \
                                     self.label[k] * (
                                             self.alpha[k] - alpha_kold) * self.K[k, k]
                                if 0 < self.alpha[i] < self.C:
                                    self.b = b1
                                elif 0 < self.alpha[k] < self.C:
                                    self.b = b2
                                else:
                                    self.b = (b1 + b2) / 2.0
                                print('{} is starting'.format(i))
                                self.E = (np.dot(self.alpha * self.label, self.K) + self.beta - self.label).ravel()
                                if abs(self.alpha[k] - alpha_kold) < 0.000001:
                                    print('{} not moving enough iter {}'.format(k,i))
            if KKT_check:
                KKT_check=False
            if not alpha_changed:
                KKT_check=True
            iter+=1
    def predict(self,x):
        x=np.array(x)
        return np.dot(self.ker_function(x),self.alpha*self.label)+self.beta
    def clipalpha(self,aj,L,H):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines(): # 对文本按行遍历
        lineArr = line.strip().split()
        dataMat .append([float(lineArr[1]), float(lineArr[2])])   # 每行前两个是属性数据，最后一个是类标号
        labelMat .append(float(lineArr[3]))
    return np.array(dataMat),np.array(labelMat)
if __name__ == '__main__':
    x,y=loadDataSet('testSetRBF.txt')
    x_test,y_test=loadDataSet('testSetRBF2.txt')
    svm=SVM(x,y,10,800)
    svm.trian()
    print(svm.alpha)
    total = len(x)
    count = 0
    for i in range(total):
        if svm.predict(x[i]) * y[i] > 0:
            count += 1
    print('the acc rate of train set is {}%'.format(count / total * 100))
    total=len(x_test)
    count=0
    for i in range(total):
        if svm.predict(x_test[i])*y_test[i]>0:
            count+=1
    print('the acc rate of test set is {}%'.format(count/total*100))