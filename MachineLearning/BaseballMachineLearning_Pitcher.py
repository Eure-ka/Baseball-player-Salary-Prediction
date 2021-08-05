#!/usr/bin/env python
# coding: utf-8

# # 세이버 스텟을 이용한 한국 프로야구 타자(Pitcher) 데이터 분석

# # [STEP 1] 탐색 : 프로야구 연봉 데이터  

# ## 라이브러리 불러오기 및 기본 설정

# In[24]:


# -*- coding : utf-8 -*- 
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Data source : http://www.statiz.co.kr/
import warnings
warnings.filterwarnings("ignore")

mpl.rc('font', family='Malgun Gothic')


# ## 데이터 가져오기

# In[25]:


pitcher = pd.read_csv( "AllPitcher.csv", encoding = 'CP949')
batter = pd.read_csv("AllBatter.csv", encoding = 'CP949')


# ## Pitcher(투수) 데이터 분석

# ## 데이터 살펴보기

# In[26]:


pitcher.columns


# In[27]:


pitcher.head()


# In[29]:


print(pitcher.shape)


# ## 예측할 대상인 '연봉'에 대한 정보 확인

# In[30]:


pitcher['SALARY(2020)'].describe()


# In[31]:


# 2020년 연봉 분포를 출력
pitcher['SALARY(2020)'].hist(bins = 100)


# In[32]:


# 연봉의 상자 그림 출력
pitcher.boxplot(column = ['SALARY(2020)'])


# In[35]:


# 회귀 분석에 사용할 Feature 살펴보기
pitcher_features_df = pitcher[['WAR', 'FIP', 'LOB', 'BABIP', 'SALARY(2018)', 'SALARY(2019)']]

# 투수 각각에 대한 히스토그램 출력
def plot_hist_each_column(df):
    plt.rcParams['figure.figsize'] = [20, 16]
    fig = plt.figure(1)
    
    # df의 열 개수 만큼의 subplot을 출력
    for i in range(len(df.columns)):
        ax = fig.add_subplot(5, 5, i+1)
        plt.hist(df[df.columns[i]], bins = 50)
        ax.set_title(df.columns[i])
    plt.show()
    
plot_hist_each_column(pitcher_features_df)


# # [STEP 2]  예측 : 투수의 연봉 예측하기

# ## Feature Scaling

# In[37]:


# 판다스 형태로 정의된 데이터 출력 시 scientific-notation이 아닌 float 모양으로 출력
pd.options.mode.chained_assignment = None

# 피처 각각에 대한 스케일링을 수행하는 함수를 정의
def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x : (x-series_mean)/series_std)
    return df

# 피처 각각에 대한 Feature Scaling - 표준화 방법 이용
scale_columns = ['WAR', 'FIP', 'LOB', 'BABIP', 'SALARY(2018)', 'SALARY(2019)']
pitcher_df = standard_scaling(pitcher, scale_columns)
pitcher_df = pitcher_df.rename(columns = {'SALARY(2020)' : 'y'})
pitcher_df.head(5)


# ## Train Set, Test Set의 분리

# In[42]:


# 회귀분석을 위한 학습, 테스트 데이터셋 분리
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# 학습 데이터와 테스트 데이터로 분리
X = pitcher_df[pitcher_df.columns.difference(['NAME', 'y'])]
y = pitcher_df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 19)


# ## 회귀 분석 계수 학습 & 학습된 계수 출력

# In[43]:


# 회귀 분석 계수 학습(회귀 모델 학습)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# 학습된 계수 출력
print(lr.coef_)


# # [STEP 3]  평가 : 예측 모델 평가하기

# In[44]:


# 가장 영향력이 강한 피처 확인 
import statsmodels.api as sm

# statsmodel 라이브러리 회귀 분석 수행
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
model.summary()


# In[45]:


# 한글 출력을 위한 사전 설정 단계
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['figure.figsize'] = [20, 16]

# 회귀 계수 리스트로 반환
coefs = model.params.tolist()
coefs_series = pd.Series(coefs)

# 변수명을 리스트로 반환
x_labels = model.params.index.tolist()

# 회귀계수 출력
ax = coefs_series.plot(kind = 'bar')
ax.set_title('feature_coef_graph')
ax.set_xlabel('x_features')
ax.set_ylabel('coef')
ax.set_xticklabels(x_labels)


# ## 예측 모델 평가하기 -  방법1) R2 score

# In[56]:


# 학습 데이터와 테스트 데이터로 분리
X = pitcher_df[pitcher_df.columns.difference(['NAME', 'y'])]
y = pitcher_df['y']
X_train, X_test, y_trian, y_test = train_test_split(X, y, test_size=0.2, random_state = 15)

# 회귀 분석 모델을 학습
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# 회귀 분석 모델을 평가
print(model.score(X_train, y_train))       # train R2 score 출력
print(model.score(X_test, y_test))         # test R2 score 출력


# ## 예측 모델 평가하기 - 방법2) RMSE score

# In[58]:


# 회귀 분석 모델 평가 
y_predictions = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, y_predictions)))           # train RMSE score 출력
y_predictions = lr.predict(X_test)
print(sqrt(mean_squared_error(y_test, y_predictions)))            # test RMSE score 출력


# ## Feature 간 상관관계 분석하기 - 히트맵 이용

# In[72]:


import seaborn as sns

# 피처 간의 상관계수 행렬 계산
corr = pitcher_df[scale_columns].corr(method='pearson')
show_cols = ['WAR', 'wRC', 'wRAA', 'wOBA', 'SALARY(2017)', 'SALARY(2018)']

# corr 행렬 히트맵을 시각화
plt.rc('font', family = 'Malgun Gothic')
sns.set(font_scale = 1.5)
hm = sns.heatmap(corr.values, 
                cbar = True,
                annot = True,
                square = True,
                fmt = '.2f',
                annot_kws = {'size':15},
                yticklabels = show_cols,
                xticklabels = show_cols)

plt.tight_layout()
plt.show()


# ## 회귀 분석 예측 성능을 높이기 위한 방법 : 다중 공산성 확인

# In[61]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# 피처마다의 VIF 계수를 출력합니다.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# # [STEP 4]  시각화 : 분석 결과 시각화하기

# ## 2020년 예측 연봉과 실제 연봉 비교하기

# In[90]:


# 2020년 연봉을 예측하여 데이터 프레임 열로 생성
X = pitcher_df[['WAR', 'FIP', 'LOB', 'BABIP', 'SALARY(2018)', 'SALARY(2019)']]
predict_2021_salary = lr.predict(X)
pitcher_df['E_SALARY(2020)'] = pd.Series(predict_2021_salary)

# 원래의 데이터 프레임 불러오기
pitcher = pd.read_csv( "AllPitcher.csv", encoding = 'CP949')
pitcher = pitcher[['NAME', 'SALARY(2019)']]

# 원래의 데이터 프레임에 2021년 정보를 합치기
result_df = pitcher_df.sort_values(by = ['y'], ascending = False)
result_df.drop(['SALARY(2020)'], axis = 1, inplace = True, errors = 'ignore')
result_df = result_df.merge(pitcher, on = ['NAME'], how = 'left')
result_df = result_df[['NAME', 'y', 'E_SALARY(2020)']]
result_df.columns = ['선수명', '실제연봉(2020)', '예측연봉(2020)']

result_df.head(10)


# In[94]:


# 선수별 연봉 정보(작년연봉, 예측연봉, 실제연봉)를 막대 그래프로 출력
mpl.rc('font', family = 'Malgun Gothic')
result_df.plot(x = '선수명', y = ['실제연봉(2020)', '예측연봉(2020)'], kind = "bar")
plt.rcParams["figure.figsize"] = (150, 100)


# In[ ]:




