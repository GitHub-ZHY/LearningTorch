# _*_ coding: utf-8 _*_
# @File : 2 CfWithItemKNN.py
# @Desc : 
# @Time : 2021/9/17 9:55 
# @Author : HanYun.
# @Version：V 1.0
# @Software: PyCharm
# @Related Links:

import random
import math
from operator import itemgetter

DEBUG = True
K = 20
N = 10
trainSet = {}
testSet = {}
path = './data/ml-100k/u.data'


# region 1、数据预处理
def load_file(path):
    """
    Read files and return rows in files.
    Args:
        path:

    Returns:

    """
    with open(path, 'r') as f:
        for _, line in enumerate(f):
            yield line.strip('\r\n')


def get_dataset(path, pivot=0.75):
    """
    Divide the dataset into train / test.
    Args:
        path:
        pivot:

    Returns:

    """
    global trainSet, testSet
    trainSet_len = 0
    testSet_len = 0
    for line in load_file(path):
        user, movie, rating, timestamp = line.split('\t')
        if (random.random() < pivot):
            trainSet.setdefault(user, {})
            trainSet[user][movie] = rating
            trainSet_len += 1
        else:
            testSet.setdefault(user, {})
            testSet[user][movie] = rating
            testSet_len += 1
    print('Split trainingSet and testSet success!')
    print(f'TrainSet = {trainSet_len}')
    print(f'testSet = {testSet_len}')


get_dataset(path)


# endregion

# region 2、模型构建
class ItemBasedKNN():
    # 初始化参数
    def __init__(self, K, N):
        # 找到相似的K部电影，为目标用户推荐N部电影
        self.n_sim_movie = K
        self.n_rec_movie = N

        # 将数据集划分为训练集和测试集
        self.trainSet = trainSet  # key -> user ; value -> { item: rating, ... }
        self.testSet = testSet

        # 用户相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie)
        print('Recommneded movie number = %d' % self.n_rec_movie)

    # 计算电影之间的相似度
    def calc_movie_sim(self):
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 1
                self.movie_popular[movie] += 1
        self.movie_count = len(self.movie_popular)
        print("Total movie number = %d" % self.movie_count)

        # { user1: { movie1: 1.0, movie2: 2.0 }, user2 .... }
        for user, movies in self.trainSet.items():  # 计算电影同时出现的次数
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    # self.movie_sim_matrix[m1].setdafault(m2, 1)
                    self.movie_sim_matrix[m1][m2] = self.movie_sim_matrix[m1].get(m2, 1)
                    self.movie_sim_matrix[m1][m2] += 1
        print("Bulid co-rated users matrix success!")

        # 计算电影之间的相似性
        print("Calculating movie similarity matrix")
        for m1, related_movies in self.movie_sim_matrix.items():
            for m2, count in related_movies.items():
                # 注意 0 向量的处理, 即某电影的用户数2为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    # 避免热门商品带来偏置
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('Calculate movie similarity matrix success!')


# endregion


# region 3、Item相似性计算
knn = ItemBasedKNN(K, N)
knn.calc_movie_sim()


# endregion

# region 4、推荐函数与评估函数的定义
def recommend(user, knn, K, N):
    """
    针对目标用户U，找到K部相似的电影，并推荐其N部电影
    Args:
        user:
        knn:
        K:
        N:

    Returns:

    """
    global trainSet
    K = K
    N = N
    rank = {}
    watched_movies = trainSet[user]

    for movie, rating in watched_movies.items():
        for related_movie, w in sorted(knn.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:  # 按字典值排序
            if related_movie in watched_movies:
                continue
            rank.setdefault(related_movie, 0)
            rank[related_movie] += w * float(rating)
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]


def evaluate(knn, K, N):
    """
    产生推荐并通过准确率、召回率进行评估
    Args:
        knn:
        K:
        N:

    Returns:

    """
    print('Evaluating start ...')
    global trainSet, testSet
    N = N
    # 准确率和召回率
    hit = 0
    rec_count = 0
    test_count = 0
    # 覆盖率
    all_rec_movies = set()

    for i, user in enumerate(trainSet):
        test_moives = testSet.get(user, {})
        rec_movies = recommend(user, knn, K, N)
        for movie, w in rec_movies:
            if movie in test_moives:
                hit += 1
            all_rec_movies.add(movie)
        rec_count += N
        test_count += len(test_moives)

    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    print('precisioin=%.4f\trecall=%.4f\t' % (precision, recall))


evaluate(knn, K, N)
# endregion
