# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import random as rd

import matplotlib.pyplot as plt
import pickle as pk
#그래프 그리기
import seaborn as sns

from sklearn import model_selection,neighbors,linear_model,tree,svm
from sklearn import preprocessing
from sklearn.cluster import KMeans
#평균값과 표준편차 구해주는 라이브러리
from sklearn.decomposition import PCA
#데이터 회귀분석 라이브러리
from statsmodels.formula.api import ols
#상관계수 구하는 라이브러리
from scipy import stats




# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)


# =============================================================================
# 분류
# =============================================================================

def get_data():
    #csv 읽어오기 'cp949'는 한글 깨짐 방지
    data = pd.read_csv("./data/AAII.csv", encoding='cp949')
    #data.info()를 해보니 습도(humidity)가 object형이길래 float형으로 바꿔주었습니다.
    ww1 = preprocessing.OrdinalEncoder()
    data["humidity_int"] = ww1.fit_transform(data[["humidity"]])
    print(data.info())
    
    x = data.iloc[:,[2,3,4,7,8,10]]
    y = data.iloc[:,9]
    
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.3)
    
    
    return x_train,x_test,y_train,y_test

get_data ()

def classfication():
    x_train,x_test,y_train,y_test = get_data()
    # model = neighbors.KNeighborsClassifier()
    #max_depth은 트리의 최대 깊이를 설정하는 파라미터입니다.
    #깊이를 7로하여 정확도를 높였습니다.
    model = tree.DecisionTreeClassifier(max_depth=7)
    # model = svm.SVC()
    
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    
    print(score)
    
    pk.dump(model,open("분류2.m","wb"))
    
# classfication()

def a_predict():
    model = pk.load(open("분류2.m","rb"))
    x = [-5.7,2.8,4.4,4.1,4.3,64]
    #-2.2,0,0,1.6,8.9,40 맑음
    #8.4,18,0,10,0,94 흐림/비
    x = np.array(x)
    x = x.reshape(1,-1)
    
    y_pre = model.predict(x)
    print(y_pre)
    
# a_predict()


# =============================================================================
# 회귀
# =============================================================================

def get_dataa():
    data = pd.read_csv("./data/RE.csv", encoding="cp949")
    ww1 = preprocessing.OrdinalEncoder()
    
    data["humidity_int"] = ww1.fit_transform(data[["humidity"]])
    print(data.info())
    # print(data.head())
    x = data.iloc[:,[2,3,5,6,8]]
    y = data.iloc[:,7]
    
    
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.3)
    
    return x_train,x_test,y_train,y_test,data
# get_dataa()


def regression():
    
    x_train,x_test,y_train,y_test,data = get_dataa()
    # model = linear_model.LinearRegression()
    #가장 정확도가 높은 알고리즘인 Tree의 최대 깊이를 7로 설정했습니다.
    model = tree.DecisionTreeRegressor(max_depth=7)
    # model = neighbors.KNeighborsRegressor()
    model.fit(x_train,y_train)
    
    score = model.score(x_test,y_test)
    print(score)
    pk.dump(model,open("회귀1.m","wb"))
    
    #신뢰구간을 포함한 그래프 그리기
    sns.lmplot(x='humidity_int',y='weather',data=data)
    
    #회귀계수와 p값 f값 구하기
    res = ols('humidity_int ~ weather', data=data).fit()
    print(res.summary())
    x = data['humidity_int']
    y = data['weather']
   
    #피어슨의 상관계수 구하기
    cor = stats.pearsonr(x, y)
    print(cor)
    
# regression()

def b_predict():
    model = pk.load(open("회귀1.m","rb"))
    x = [18,0,10,0,94]
    # [18,0,10,0,94] 흐림/
    # 0,0,0,9.8,40] 맑음
    x = np.array(x)
    x = x.reshape(1,-1)
    y_pre = model.predict(x)
    
    tmp=""
    if y_pre<=15.8:
        tmp="맑음,구름조금"
    elif y_pre>15.8:
        tmp = "습도높은맑음,흐림,구름,비,눈"
    print(y_pre,tmp,"//예외가 있을 수 있습니다.")
    

# b_predict()
    

# =============================================================================
# 클러스터링
# =============================================================================

def get_cluster():
    data = pd.read_csv("./data/AAII.csv", encoding='cp949')
    ww1 = preprocessing.OrdinalEncoder()
    
    data["humidity_int"] = ww1.fit_transform(data[["humidity"]])
    print(data.info())
    
    x = data.iloc[:,[3,4,7,8]]
    y = data.iloc[:,9]
    
    #값을 array로 넘겨줍니다.
    return x.values,y


# get_cluster()
    

def cluster():
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    x,y = get_cluster()
    
    x_data = pd.DataFrame(data=x, columns=['rain','snow','cloud','sun'])
    
    #n_clusters는 그룹의 개수
    # kmeans++은 중심점을 조금 더 신중히 결정하는 kmeans의 단점을 보안한 것
    # max_iter은 최대 반복 횟수
    # random_state 무작위 값 0
    kmeans = KMeans(n_clusters = 5, init='k-means++',max_iter=300,random_state=0)
    kmeans.fit(x_data)
    print(kmeans.labels_)
    # for i in kmeans.labels_:
    #     print(i,y[i])
        
    x_data['cluster'] = kmeans.labels_
    x_result = x_data.groupby(['cluster'])['rain'].count()
    print(x_result)
    
   # 평균과 표준편차를 구하기(점찍기 위해서)
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(x)
    
    x_data['pca_x'] = pca_transformed[:,0]
    x_data['pca_y'] = pca_transformed[:,1]
    
    print(x_data.head(300))

    #cluster의 값이 0일때 mark_0 밑은 이와 같습니다.
    mark_0 = x_data[x_data['cluster']==0].index
    mark_1 = x_data[x_data['cluster']==1].index
    mark_2 = x_data[x_data['cluster']==2].index
    mark_3 = x_data[x_data['cluster']==3].index
    mark_4 = x_data[x_data['cluster']==4].index
    
    #pca값에 따라 점을 찍습니다.
    # plt.scatter(x=x_data.loc[mark_0,'pca_x'], y=x_data.loc[mark_0,'pca_y'],marker='o',label='구름많음, 흐림, 약간의 눈,비')
    # plt.scatter(x=x_data.loc[mark_1,'pca_x'], y=x_data.loc[mark_1,'pca_y'],marker='s',label="강수량(50mm)이상")
    # plt.scatter(x=x_data.loc[mark_2,'pca_x'], y=x_data.loc[mark_2,'pca_y'],marker='^',label="맑음, 구름약간")
    # plt.scatter(x=x_data.loc[mark_3,'pca_x'], y=x_data.loc[mark_3,'pca_y'],marker='v',label="강수량(100mm)이상")
    # plt.scatter(x=x_data.loc[mark_4,'pca_x'], y=x_data.loc[mark_4,'pca_y'],marker='8',label='구름많음, 많은 비')
    
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # #라벨 이름 넣기
    # plt.legend()
    # plt.show()  
    
    # 최적의 k값 구하기
    score=[]
    for k in range(1,15):
        km = KMeans(n_clusters=k)
        km.fit(x)
        #inertia는 자신의 군집까지의 거리
        score.append(km.inertia_)
    print(score)
    plt.plot(score)
    plt.show()
    
    

cluster()
    
    
















