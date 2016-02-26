#!/usr/bin/env python
"""The file contains the custom-built wrappers for the rossmann sales challenge"""
__author__ = "Agoston Torok, Krisztian Varga, and Balazs Feher"
__copyright__ = "Copyright 2016, YetAnotherDirtyKeltaGod team"
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Agoston Torok"
__email__ = "torokagoston@gmail.com"
__status__ = "Prototype"


import sys
import pickle
import time
import logging
import itertools
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print('Rossmann Helper Functions imported')

def plot_feature_importances(model, features):
    plt.figure(figsize = (10,15))
    features = np.array(features)
    indices = np.argsort(model.feature_importances_)
    plt.barh(np.arange(len(indices))-.4,model.feature_importances_[indices],color = 'k')
    plt.yticks(np.arange(len(indices)),features[indices])
    plt.title('Relative importance of features')
    plt.ylim(-0.5, len(model.feature_importances_))

def plot_results(model, X_train, X_test, y_train, y_test, savemodel = True):
    #plot training data
    plt.figure(figsize = (20,8))
    plt.subplot(121)
    yhat = model.predict(X_train)
    plt.scatter(y_train, yhat,alpha = 0.01, c = 'k')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.xlim(plt.ylim(min(y_train.values),max(y_train.values)))
    plt.plot(plt.xlim(), plt.ylim(), '--')
    plt.title('Fit on the training data')

    #plot test data
    plt.subplot(122)
    yhat = model.predict(X_test)
    plt.scatter(y_test,yhat,alpha = 0.01, c = 'k')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.xlim(plt.ylim(min(y_test.values),max(y_test.values)))
    plt.plot(plt.xlim(), plt.ylim(), '--')
    loss_value = np.sqrt(np.mean(((y_test - yhat)/y_test) ** 2))
    mytitle = 'The error of this model is{}'.format(loss_value)
    plt.title(mytitle)
    
    #savemodel = query_yes_no("Do you want to save this model?") # not working in ipython at the moment
    if savemodel:
        file_name = time.strftime("%m_%d_%Y_%Hhrs%Mmins_") + str(loss_value) + '.pkl'
        print('Saving model as {}'.format(file_name))
        fileObject = open(file_name, 'wb')
        pickle.dump(model, fileObject)
        fileObject.close()

def cluster_all_stores(table, cut_thresh = 0.05, method = 'ward'):
    Y = cdist(table.T.fillna(0),table.T.fillna(0), 'euclidean') #correlation based could be an alternative
    Z = sch.linkage(Y, method = "ward")
    plt.figure(figsize = (20,10))
    plt.subplot(1,3,(1,2))
    dend = sch.dendrogram(Z, color_threshold = cut_thresh*max(Z[:,2]), orientation = 'right',
                          no_labels = True)
    labels = sch.fcluster(Z, cut_thresh*max(Z[:,2]), criterion='distance', depth=2, R=None, monocrit=None)
    plt.subplot(1,3,3)
    pd.Series(labels).value_counts(sort = False).plot('barh', title = 'Number of store in the given group', color = 'grey')
    return labels

# Before start modeling let's convert categorical columns to dummy columns
def create_dummy_columns(df_original, columns):
    df = df_original.copy()
    for i in range(len(columns)):
        df.loc[:,columns[i]] = df.loc[:,columns[i]].astype('int').astype('str') + '_' + str(columns[i])
        df = pd.concat([df, 
                        pd.get_dummies(df.loc[:,columns[i]])], axis = 1)
    return df


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [Y/N] "
    elif default == "yes":
        prompt = " [Y/N default is YES] "
    elif default == "no":
        prompt = " [Y/N default is NO] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


# --- Krisztian ---
def addTimeToClose(df):
    #feed the 'open' data to itertools as a series
    condition = df.loc[:,'Open']
    #list comprehension and calculate the number of consecutive entries of the same kind
    mylist = [ [len(list(group)),key] for key, group in itertools.groupby( condition ) ]
  
    #lets add this to a temporary dataframe
    temp = pd.DataFrame(np.array([item for sublist in mylist for item in sublist]).reshape(-1,2))
    #Because we are interested in the size of the next close let's create a cell serving that purpose
    temp['Future'] = temp.copy().shift().loc[:,0]
    temp.loc[0,'Future'] = 0
    #And on other in the other direction to create the past
    temp['Past'] = temp.copy().shift(-1).loc[:,0]
    lastindex = temp.index.values[-1]
    temp.loc[lastindex,'Past'] = 0

    temp['DaysLeft'] = temp.loc[:,0].apply(lambda x: list(reversed(np.arange(x)+1)))
    temp['LengthNextClose'] = temp.loc[:,0].apply(lambda x: np.repeat(1,x))*temp.loc[:,'Future']
    temp['DaysAfter'] = temp.loc[:,0].apply(lambda x: list(np.arange(x)+1))
    temp['LengthLastClose'] = temp.loc[:,0].apply(lambda x: np.repeat(1,x))*temp.loc[:,'Past']

    # comprehend these and tp

    df['DaysLeft'] = [item for sublist in temp.DaysLeft.values.tolist() for item in sublist]
    df['LengthNextClose'] = [item for sublist in temp.LengthNextClose.values.tolist() for item in sublist]
    df['DaysAfter'] = [item for sublist in temp.DaysAfter.values.tolist() for item in sublist]
    df['LengthLastClose'] = [item for sublist in temp.LengthLastClose.values.tolist() for item in sublist]

    if 'Sales' in df:
        df.loc[df.Open == 1].loc[df.DaysLeft < 30].loc[df.LengthNextClose > 10].groupby('DaysLeft').mean().loc[:,'Sales'].plot()
        df.loc[df.Open == 1].loc[df.DaysAfter < 30].loc[df.LengthLastClose > 10].groupby('DaysAfter').mean().loc[:,'Sales'].plot()
    
    # Filter out data for short close intervals, also include only the first 6 days before/after the event
    df.loc[df['LengthNextClose'] < 10, 'DaysLeft'] = 0
    df.loc[df['LengthLastClose'] < 10, 'DaysAfter'] = 0
    df.loc[df['DaysLeft'] > 6, 'DaysLeft'] = 0
    df.loc[df['DaysAfter'] > 6, 'DaysAfter'] = 0
    df.drop('LengthNextClose', axis=1, inplace=True)
    df.drop('LengthLastClose', axis=1, inplace=True)

    # Bare eye check
    if 'Sales' in df:
        pass
        #df.loc[df.Open == 1].loc[df.DaysLeft < 30].groupby('DaysLeft').mean().loc[:,'Sales'].plot()
        #df.loc[df.Open == 1].loc[df.DaysAfter < 30].groupby('DaysAfter').mean().loc[:,'Sales'].plot()
		
def ext_load_currency_data(path):
    ext_curr = pd.read_csv(path, parse_dates=[0])
    # Fill missing values with the last known day's value
    valuedict = dict()
    res = []
    for _, row in ext_curr.iterrows():
        valuedict[row['Date']] = row['Rate']
    lastknown = 1
    for d in pd.date_range(pd.datetime(2013, 1, 1), pd.datetime(2015, 12, 31)):
        if d in valuedict:
            lastknown = valuedict[d]
        res.append({'Date': d, 'EurUsdRate' : lastknown})
    return pd.DataFrame(res)

def ext_load_weather_data(path):
    ext_weather = pd.read_csv(path, parse_dates=[0], converters={'': str})
    temp = { row[0] : row[1:] for row in ext_weather.values }
    def countPhenomenon(v, s):
        return sum(1 for _ in (filter(lambda x: s in x, v)))
    res = []
    #Calculate ratio for different phenomenons for each day
    for k,v in temp.items():
        v = [str(x).lower() for x in v] # just to make sure
        # 'nan' means sunny (empty string in csv converts to nan), 'unknown' means data was n/a on that day
        d = { 'Date': k }
        valid = len(v) - countPhenomenon(v, 'unknown')
        #for p in ['rain','hail', 'fog', 'thunderstorm', 'snow', 'tornado']:
        #    d["Weather" + p.title()] = countPhenomenon(v, p) / (valid * 1.0)
        d["WeatherBad"] = sum(1 for _ in ( filter(lambda x: any([pheno in x for pheno in ['rain', 'hail', 'thunderstorm', 'snow', 'tornado']]), v))) / (valid * 1.0)
        res.append(d)
    return pd.DataFrame(res)
    
def ext_load_search_data(path):
    ext_search = pd.read_csv(path, parse_dates=[0,1])
    res = []
    #Convert weekly averages to redundant daily entries for join
    for _, row in ext_search.iterrows():
        for day in pd.date_range(row['from'], row['to']):
            res.append({'Date': day, 'SearchRossmanGermany': row['rossmann-ger'], 'SearchRossmanWorld': row['rossmann-ww']})
    return pd.DataFrame(res)

def longest_streak(arr, val):
    maxv = 0
    curr = 0
    for v in arr:
        if v == val:
            curr += 1
        else:
            if curr > maxv: 
                maxv = curr
            curr = 0
               
    if curr > maxv: 
        maxv = curr
        
    return maxv

# Gather some features
def build_features(data_in):
    data = data_in.copy()
    data.fillna(0, inplace=True)
    
    # Label encode some features
    # these are important for regression not for tree classification
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    # Promo open time in months
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1
    #drop intermediate steps for memory sake
    data.drop('monthStr', axis=1, inplace=True)
    return data
