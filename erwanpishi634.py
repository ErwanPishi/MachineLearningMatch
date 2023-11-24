import numpy as np
import pandas as pd
import seaborn as sb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,StratifiedKFold,train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error,roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

#%%处理 x2
def x2process(x, x2_index): 
    x2 = x.copy()

    x2['B1'] = x2['B1'].str[0:4].astype(float)-3468
      
    tmp_dict = {'A':1,
                'B':2,
                'C':3, 
                'D':4,
                'E':5,
                'F':6,
                'G':7}     
    x2['B3'] =  x2['B3'].map(tmp_dict)
    
    x2['B4'] = x2['B4'].str[1:].astype(float)
    x2['B6'] = x2['B6'].str[1:]
    
    x2['B8'] = x2['B8'].str[0:4].astype(float)
    x2['B8'].fillna(3835,inplace=True)
    x2['B8'] = x2['B8']-3469 
    
    x2['B15'] = x2['B15'].str[1:]
    x2['B15'] = x2['B15'].astype(float)
    
    x2['B16'] = x2['B16'].str[1:]
    x2['B16'] = x2['B16'].astype(float)
    
    x2['B19'] = x2['B19'].str[1:].astype(float)   
    
    x2['B20'] = x2['B8']-x2['B2']
    x2['货单时长'] = x2['B8']-x2['B1']
    x2['货单时长月'] = (x2['货单时长']+1)%12+1
    x2['货单开始月'] = x2['B1']%12+1
    x2['货单结束月'] = x2['B8']%12+1
    x2['B21'] = x2['B8']-x2['B1']
    x2['货单是否提前结束'] = x2['B21'].apply(lambda x: 0 if x>=0 else 1)
    
    x2_columns = ['B1','B2','B5','B7','B8','B9','B10',
                  'B11','B13','B14','B17','B20']
    
    x2_mean = pd.DataFrame(data=np.zeros((x2_index.shape[0],len(x2_columns))), 
                      index=x2_index,columns=x2_columns)
    x2_min = pd.DataFrame(data=np.zeros((x2_index.shape[0],len(x2_columns))), 
                      index=x2_index,columns=x2_columns)
    x2_max = pd.DataFrame(data=np.zeros((x2_index.shape[0],len(x2_columns))), 
                      index=x2_index,columns=x2_columns)
    x2_range = pd.DataFrame(data=np.zeros((x2_index.shape[0],len(x2_columns))), 
                      index=x2_index,columns=x2_columns)
    x2_sum = pd.DataFrame(data=np.zeros((x2_index.shape[0],len(x2_columns))), 
                      index=x2_index,columns=x2_columns)
    
    
    grouped = x2.groupby('客户编号')
    for c in x2_columns:
        x2_mean[c] = grouped[c].mean().to_frame()   
        x2_min[c] = grouped[c].min().to_frame()
        x2_max[c] = grouped[c].max().to_frame()
        x2_range[c] = x2_max[c]-x2_min[c]
        x2_sum[c] = grouped[c].sum().to_frame()
        
    
    x2_mean.columns =[i+'mean' for i in x2_columns]
    x2_min.columns = [i+'min' for i in x2_columns]
    x2_max.columns = [i+'max' for i in x2_columns]
    x2_range.columns=[i+'range' for i in x2_columns]
    x2_sum.columns=[i+'sum' for i in x2_columns]
    

    new_x2 = pd.merge(x2_mean,x2_min,on='客户编号')
    new_x2 = pd.merge(new_x2,x2_max,on='客户编号')
    new_x2 = pd.merge(new_x2,x2_range,on='客户编号')
    new_x2 = pd.merge(new_x2,x2_sum,on='客户编号')
   
    
    #衍生变量：货单总数
    new_x2['total_buy_cnt'] = grouped['B1'].count() #用B1计数 单纯是因为他没有空值
    
    #衍生变量：购买月份数
    new_x2['buy_mth_cnt'] = x2['货单开始月'].nunique()
    
    #衍生变量：月均购买次数（只算购买月份）
    new_x2['buy_cnt_per_mth'] = new_x2['total_buy_cnt']/new_x2['buy_mth_cnt']
    
    #有些货单结束日期比货单开始日期提前一天 不知道是什么原因
    #衍生变量：提前结束数量、比例
    new_x2['commodity_adv_cnt'] = grouped['货单是否提前结束'].sum()
    new_x2['commodity_adv_ratio'] = grouped['货单是否提前结束'].sum()/new_x2['total_buy_cnt']
   
    #有货款总金额和应付货款总金额
    #衍生变量：已付货款总金额、比例
    new_x2['paid_total_amount'] = grouped['B10'].sum()-grouped['B11'].sum()
    new_x2['paid_total_amount_ratio'] = new_x2['paid_total_amount']/grouped['B10'].sum()
    
    #最终未选用std，因为效果变差
    '''
    x2_std = pd.DataFrame(data=np.zeros((x2_index.shape[0],len(x2_columns))), 
                      index=x2_index,columns=x2_columns)
    '''
    return new_x2

x1_train = pd.read_csv('X1_train.csv',header=[0], index_col=0)
x1_test = pd.read_csv('X1_test.csv',header=[0], index_col=0)

x2_train = pd.read_csv('X2_train.csv',header=[0], index_col=0).dropna(how='all',axis=1)
new_x2_train = x2process(x2_train, x1_train.index)

x2_test = pd.read_csv('X2_test.csv',header=[0], index_col=0)
new_x2_test = x2process(x2_test, x1_test.index)

#%% 处理 x3
def x3process(x3):
    x3 = x3[['C2','C3', 'C4', 'C5','C8', 'C10', 'C13', 'C14', 'C16', 'C18', 'C20', 'C29', 
             'C30', 'C32', 'C34', 'C36', 'C43', 'C46', 'C47']]
    temp = list(x3['C2'])
    for i in range(0, len(temp)):
        if temp[i] == -1.2475837768427405:
            temp[i] = 1
        else: temp[i] = 0
    temp = pd.Series(temp)
    temp.index=x3.index
    x3['C2'] = temp.astype(str).astype(float)
    for i in range(x3.shape[1]):
        x3.iloc[:,i] = x3.iloc[:,i].fillna(x3.iloc[:,i].mean())
    x3 = x3[['C10','C13','C14','C18']]
    return x3

x3_train = pd.read_csv('X3_train.csv',header=[0], index_col=0)
new_x3_train = x3process(x3_train)

x3_test = pd.read_csv('X3_test.csv',header=[0], index_col=0)
new_x3_test = x3process(x3_test)
#%%
cols = []
for i in ['B2','B7','B9']:
    for j in ['max','range','sum','mean']:
        cols.append(i+j)
'''
for i in ['B1']: 这个如果删了反而下降了 惹不起
    for j in ['sum','mean']:
        cols.append(i+j)
'''  
x_train = pd.merge(new_x2_train, new_x3_train,on='客户编号')
x_train.drop(columns=cols,inplace=True)

x_test = pd.merge(new_x2_test, new_x3_test,on='客户编号')
x_test.drop(columns=cols,inplace=True)

y_train = pd.read_csv('y_train.csv',header=[0], index_col=0)['复购频率']


#%% lgb
#特征工程前 线下0.6304 而线上0.6460
#特征工程后(只保留衍生特征total_buy_cnt) 线下0.6280
#特征工程后(全部衍生特征均保留) 线下0.6285 而线上0.6458

# 设定xgb参数
params={
    'objective':'binary:logistic'
    ,'eval_metric':'auc'
    ,'n_estimators':500
    ,'eta':0.03
    ,'max_depth':3
    ,'min_child_weight':100
    ,'scale_pos_weight':1
    ,'gamma':5
    ,'reg_alpha':10
    ,'reg_lambda':10
    ,'subsample':0.7
    ,'colsample_bytree':0.7
    ,'seed':123
}


ys_train = y_train.copy()
for i in ys_train.index:
    if ys_train[i] == 2:
        ys_train[i] = 3
        
import lightgbm as lgb
import catboost as cat

rng = np.random.RandomState(31337)
kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=rng)
correct_sum = 0
total_sum = 0
for train_index,test_index in kf.split(x_train,ys_train):
    oof_preds = np.zeros(len(test_index))
    
    lgb_model = lgb.LGBMRegressor(n_estimators=1000,
                                 learning_rate=0.01, 
                                 max_depth=4)
    
    lgb_model.fit(x_train.iloc[train_index], 
                  ys_train.iloc[train_index])
    '''
    cat_model = cat.CatBoostRegressor(iterations=1000,
                                      learning_rate=0.02,
                                      depth=3,
                                      loss_function ='RMSE',
                                      eval_metric='RMSE',
                                      od_wait=200,
                                      silent=True
                                      )
    cat_model.fit(x_train.iloc[train_index], 
                  ys_train.iloc[train_index])
    '''
    predictions = lgb_model.predict(x_train.iloc[test_index])
    actuals = y_train.iloc[test_index]
    
    for i in range(len(predictions)):
        if predictions[i] > 1.3:
            predictions[i] =2
        elif predictions[i] > 0.4:
            predictions[i] = 1
        else:
            predictions[i] = 0
    
    print(confusion_matrix(actuals, predictions))
    
    correct_sum += 1 * confusion_matrix(actuals, predictions)[0,0]
    correct_sum += 3 * confusion_matrix(actuals, predictions)[1,1]
    correct_sum += 5 * confusion_matrix(actuals, predictions)[2,2]

    total_sum += 1 * sum(actuals==0)
    total_sum += 3 * sum(actuals==1)
    total_sum += 5 * sum(actuals==2)

print('回归树', correct_sum/total_sum)

#%%
lgb_model = lgb.LGBMRegressor(n_estimators=1000,
                             learning_rate=0.01, 
                             max_depth=3,
                             importance_type = 'gain').fit(x_train, ys_train)

y_test_p = lgb_model.predict(x_test)
     
for i in range(len(y_test_p)):
    if y_test_p[i] > 1.3:
        y_test_p[i] =2
    elif y_test_p[i] > 0.4:
        y_test_p[i] = 1
    else:
        y_test_p[i] = 0        


#输出结果
result = pd.DataFrame(y_test_p,columns=['复购频率'])
result.to_excel(r'output/result.xlsx',index=0)
result.head()




