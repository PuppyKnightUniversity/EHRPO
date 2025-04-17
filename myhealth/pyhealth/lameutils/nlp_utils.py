# -*- coding:utf-8 -*-
import numpy as np
import re
import datetime


class NlpUtility(object):
    @staticmethod
    def is_chinese_char(x):
        if u'\u4e00' <= x <= u'\u9fff':
            return True
        return False

    @staticmethod
    def is_number(num):
        pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
        result = pattern.match(num)
        if result:
            return True
        else:
            return False

    @staticmethod
    def has_chinese_char(s):
        for i in s:
            if NlpUtility.is_chinese_char(i):
                return True
        return False

    @staticmethod
    def has_digit(s):
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        for c in s:
            if c in digits:
                return True
        return False

    @staticmethod
    def has_abc(s):
        chs = "".join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z'])
        chs_up = chs.upper()
        for c in s:
            if c in chs + chs_up:
                return True
        return False

    @staticmethod
    def all_digit(s):
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        for c in s:
            if c not in digits:
                return False
        return True

    @staticmethod
    def all_abc(s):
        chs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']
        for c in s:
            if c not in chs:
                return False
        return True

    @staticmethod
    def strtime2datetime(time):
        # print time, type(time)
        return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

    @staticmethod
    def strtime_minus(time1, time2):
        return (NlpUtility.strtime2datetime(time1) - NlpUtility.strtime2datetime(time2)).days


class DIS_JOIN(object):
    def __init__(self):
        '''
        并查集算法
        拥有两个函数
        一个是把某个元素放在某个集合中
        另一个是返回一个list，包含所有集合和集合中所有的点
        '''
        self.Set = None
        self.Sum = None
        self.n = None

    def clear(self, n):
        '''
        初始化
        n为一共有多少个元素
        '''
        self.Set = np.zeros(n, dtype = int)
        self.Set = self.Set - 1
        self.Sum = n
        self.n = n
        return

    def find_r(self, p):
        '''
        返回p属于那一个集合
        '''
        if self.Set[p] < 0:
            return p

        self.Set[p] = self.find_r(self.Set[p])
        return self.Set[p]

    def join(self, a, b):
        '''
        将元素b加入元素a所在的集合中
        '''
        ra = self.find_r(a)
        rb = self.find_r(b)
        if (ra != rb):
            self.Set[ra] = rb
            self.Sum = self.Sum - 1
        return

    def get_set(self):
        '''
        返回一个list，每个元素是一个set
        '''
        Set = [None] * self.Sum
        cnt = 0

        lis = [None] * self.n
        for k in range(self.n):
            lis[k] = [k, self.find_r(k)]
        lis.sort(key=lambda x:(x[-1]))
        lis.append([-100,-100]) #加一个终止条件

        pre = lis[0][-1]
        pre_poi = 0

        for k in range(1, self.n + 1):
            if lis[k][-1] != pre:
                element_set = [None] * (k - pre_poi)
                for i in range(pre_poi, k):
                    element_set[i - pre_poi] = lis[i][0]
                Set[cnt] = np.array(element_set)
                cnt = cnt + 1
                pre = lis[k][-1]
                pre_poi = k

        return Set




# if __name__ == '__main__':
#     dis_join = DIS_JOIN()
#     dis_join.clear(5)
#     dis_join.join(0,3)
#     dis_join.join(1,4)
#     Set = dis_join.get_set()
#     print(Set)
#     # for k in range(5):
#     #     print(k, dis_join.find_r(k))
#     # print(dis_join.Sum)