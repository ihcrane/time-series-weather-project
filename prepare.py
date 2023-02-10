import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer


def train_val_test(df, target=None, stratify=None, seed=42):
    
    '''Split data into train, validate, and test subsets with 60/20/20 ratio'''
    
    train, val_test = train_test_split(df, train_size=0.6, random_state=seed)
    
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    return train, val, test
    

def x_y_split(df, target, seed=42):
    
    '''
    This function is used to split train, val, test into X_train, y_train, X_val, y_val, X_train, y_test
    '''
    
    train, val, test = train_val_test(df, target, seed)
    
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_val = val.drop(columns=[target])
    y_val = val[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test




def prep_titanic(titanic):
    
    '''
    This function is used to drop unnecessary columns for the titanic data 
    and create dummies for the sec and embark_town columns
    '''
    
    titanic.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age', 'Unnamed: 0'], inplace=True)
    
    titanic_dummies = pd.get_dummies(titanic[['sex', 'embark_town']], drop_first=True)
    titanic = pd.concat([titanic, titanic_dummies], axis=1)
    
    return titanic



def prep_telco(telco):
    
    '''
    This function is used to drop unnecessary columns, convert the total_charges column to a float
    and create dummies for the object columns for better data manipulation later
    '''
    
    telco.drop(columns=['Unnamed: 0', 'payment_type_id', 'contract_type_id', 
                        'internet_service_type_id', 'customer_id'], inplace=True)
    
    telco['total_charges'] = (telco['total_charges'] + '0').astype('float')

    
    telco_dummies = pd.get_dummies(telco[['gender', 'partner', 'dependents', 
                                      'phone_service', 'multiple_lines', 
                                      'online_security', 'online_backup', 
                                      'device_protection', 'tech_support', 
                                      'streaming_tv', 'streaming_movies', 
                                      'paperless_billing', 'churn', 'internet_service_type', 
                                      'contract_type', 'payment_type']], drop_first=True)
    
    telco = pd.concat([telco, telco_dummies], axis=1)
    
    return telco




def prep_iris(iris):
    
    '''
    This function is used to drop unecessary columns, rename the species_name column
    and created dummies for the species column
    '''
    
    iris.drop(columns=['species_id', 'measurement_id', 'Unnamed: 0'], inplace=True)
    
    iris.rename(columns={'species_name':'species'}, inplace=True)
    
    iris_dummies = pd.get_dummies(iris[['species']], drop_first=True)
    iris = pd.concat([iris, iris_dummies], axis=1)
    
    return iris



def prep_zillow(df):

    '''
    This function is used to drop unecessary columns, rename the columns, drop nulls and duplicates.
    '''

    # renaming columns
    df = df.rename(columns={'bedroomcnt':'bed',
                        'bathroomcnt':'bath',
                        'calculatedfinishedsquarefeet':'sqft',
                        'taxvaluedollarcnt':'tax_value',
                        'yearbuilt':'year'})
    
    # drop Unnamed: 0 column
    df = df.drop(columns=['Unnamed: 0'])
    
    #drop nulls
    df = df.dropna()
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def mm_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Min Max Scaler and out puts
    scaled data for all three data splits
    '''
    # calls the Min Max Scaler function and fits to train data
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(train[col_list])
    
    # transforms all three data sets
    train[col_list] = mm_scaler.transform(train[col_list])
    val[col_list] = mm_scaler.transform(val[col_list])
    test[col_list] = mm_scaler.transform(test[col_list])
    
    return train, val, test

def ss_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Standard Scaler and out puts
    scaled data for all three data splits
    '''

    # calls Standard Scaler function and fits to train data
    ss_scale = StandardScaler()
    ss_scale.fit(train[col_list])
    
    # transforms all three data sets
    train[col_list] = ss_scale.transform(train[col_list])
    val[col_list] = ss_scale.transform(val[col_list])
    test[col_list] = ss_scale.transform(test[col_list])
    
    return train, val, test

def rs_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Robust Scaler and out puts
    scaled data for all three data splits
    '''

    # calls Robust Scaler funtion and fits to train data set
    rs_scale = RobustScaler()
    rs_scale.fit(train[col_list])
    
    # transforms all three data sets
    train[col_list] = rs_scale.transform(train[col_list])
    val[col_list] = rs_scale.transform(val[col_list])
    test[col_list] = rs_scale.transform(test[col_list])
    
    return train, val, test

def qt_scaler(train, val, test, col_list, dist='normal'):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Quantile Transformer and out puts
    scaled data for all three data splits
    '''

    # calls Quantile Transformer function and fits to train data set
    qt_scale = QuantileTransformer(output_distribution=dist, random_state=42)
    qt_scale.fit(train[col_list])
    
    # transforms all three data sets
    train[col_list] = qt_scale.transform(train[col_list])
    val[col_list] = qt_scale.transform(val[col_list])
    test[col_list] = qt_scale.transform(test[col_list])
    
    return train, val, test


def remove_outliers(df, num=8, k=1.5):

    '''
    This function is to remove the data above the upper fence and below the lower fence for each column.
    This removes all data deemed as an outlier and returns more accurate data. It ignores columns that 
    are categorical and only removes data for continuous columns.
    '''
    a=[]
    b=[]
    fences=[a, b]
    features= []
    col_list = []
    i=0
    for col in df:
            new_df=np.where(df[col].nunique()>num, True, False)
            if new_df:
                if df[col].dtype == 'float' or df[col].dtype == 'int':

                    # for each feature find the first and third quartile
                    q1, q3 = df[col].quantile([.25, .75])

                    # calculate inter quartile range
                    iqr = q3 - q1

                    # calculate the upper and lower fence
                    upper_fence = q3 + (k * iqr)
                    lower_fence = q1 - (k * iqr)

                    # appending the upper and lower fences to lists
                    a.append(upper_fence)
                    b.append(lower_fence)

                    # appending the feature names to a list
                    features.append(col)

                    # assigning the fences and feature names to a dataframe
                    var_fences= pd.DataFrame(fences, columns=features, index=['upper_fence', 'lower_fence'])
                    
                    col_list.append(col)
                else:
                    print(f'{col} is not a float or int')
            else:
                print(f'{col} column ignored')

    # for loop used to remove the data deemed unecessary 
    for col in col_list:
        df = df[(df[col]<= a[i]) & (df[col]>= b[i])]
        i+=1
    return df, var_fences

def plot_categorical_and_continuous_vars(train, cont_vars, cat_vars):
    train_corr = train[cont_vars].corr(method='spearman')
    sns.heatmap(train_corr)
    plt.show()
    
    for col in cont_vars:
        sns.lmplot(x='tax_value', y=col, data=train.sample(1000))
        plt.show()
    
    for col in cat_vars:
        sns.stripplot(x=col, y='tax_value', data=train.sample(1000))
        plt.show()


def plot_variable_pairs(train):
    sns.pairplot(data=train.sample(2000), diag_kind='hist', kind='reg')
    plt.show()
    