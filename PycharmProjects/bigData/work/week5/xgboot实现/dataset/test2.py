import xgboost as xgb
# read in data
import  csv
import  pandas as pd
# 读取数据。
house_info = pd.read_csv('xrp.csv')
#print (house_info[['Close','Volume','MarketCap']])

three=house_info[['Close','Volume','MarketCap']]
#print (house_info[6][6])







#
# dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
# dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# # specify parameters via map
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)