# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:42:26 2020

@author: parva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')

des = train_data.describe()

# train_data['Sex'] = pd.get_dummies(train_data['Sex'])

train_data['Fare'].isna().sum()

train_data = train_data.dropna(subset = ['Embarked'])

train_data.hist(bins=50, figsize=(20,15))
plt.show()




train_data = train_data.drop('PassengerId',axis = 1)
train_data = train_data.drop('Ticket',axis = 1)
train_data = train_data.drop('Cabin',axis = 1)

train_data_labels = train_data['Survived']
train_data = train_data.drop('Survived',axis = 1)

train_data.info()

s = train_data['SibSp'] + 1


from sklearn.base import BaseEstimator, TransformerMixin
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, CleanRareTitle = True): # no *args or **kargs
        self.CleanRareTitle= CleanRareTitle
        
    def fit(self,X,y = None):
        return self
    def transform(self,X):
        
            
        FamilySize = X['SibSp'] + X['Parch'] + 1
        FamilySize = FamilySize.to_numpy()
        IsAlone = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            if FamilySize[i] > 1:
                IsAlone[i] = 0
            
        Title = X['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        if self.CleanRareTitle:
            title_names = (Title.value_counts() < 10)
            Title = Title.apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
            
        data_extra_attribs = np.c_[X,FamilySize,IsAlone,Title]
        return data_extra_attribs
        
        # return pd.DataFrame(
        #         data_extra_attribs,
        #         columns=list(X.columns)+["FamilySize", "IsAlone", "Title"],
        #         index=X.index)
        # if self.CleanRareTitle:
        #     title_names = (data1['Title'].value_counts() < stat_min)
    
    
    
attr_adder = CombinedAttributesAdder(CleanRareTitle=True)
data_extra_attribs = attr_adder.transform(train_data)





data_extra_attribs= pd.DataFrame(
    data_extra_attribs,
    columns=list(train_data.columns)+["FamilySize", "IsAlone", "Title"],
    index=train_data.index)

# data_extra_attribs['Title'].value_counts()

data_num = train_data.drop(['Name','Sex','Embarked'],axis = 1 )
# data_num = data_num.astype(float)

from sklearn.impute import SimpleImputer


# imputer1 = SimpleImputer(strategy="median")
# imputer1.fit(data_num)
# X1 = imputer1.transform(data_num)
# data_tr1 = pd.DataFrame(X1, columns=data_num.columns,
#                           index=train_data.index)


# imputer2 = SimpleImputer(strategy="mean")
# imputer2.fit(data_num)
# X2 = imputer2.transform(data_num)
# data_tr2 = pd.DataFrame(X2, columns=data_num.columns,
#                           index=train_data.index)


# imputer3 = SimpleImputer(strategy="most_frequent")
# imputer3.fit(data_num)
# X3 = imputer3.transform(data_num)
# data_tr3 = pd.DataFrame(X3, columns=data_num.columns,
#                           index=train_data.index)


# data_tr4 = data_num.dropna()



data_cat = train_data[['Name','Sex','Embarked']]

from sklearn.preprocessing import OneHotEncoder
# cat_encoder = OneHotEncoder(sparse = False)
# data_cat_1hot = cat_encoder.fit_transform(data_cat)


# cat_encoder.categories_


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

atrr_pipeline = Pipeline([('AddAttribute',CombinedAttributesAdder(CleanRareTitle = True))])
data_trans = atrr_pipeline.fit_transform(train_data)


num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median"))])
        # ('std_scaler', StandardScaler()),
        # ('attribute_adder',CombinedAttributesAdder(CleanRareTitle = True,numerical = True )),
    
data_num_tr = num_pipeline.fit_transform(data_num)


from sklearn.compose import ColumnTransformer

trans_attribs = list(train_data)
num_attribs = list(data_num)
cat_attribs = ["Sex","Embarked"]

full_pipeline = ColumnTransformer([
        ("trans",atrr_pipeline,trans_attribs),
        ("num", num_pipeline, num_attribs),
        # ("trans",CombinedAttributesAdder(CleanRareTitle = True,numerical = False ),trans_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

data_prepared = full_pipeline.fit_transform(train_data)

# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier()
# model.fit(data_prepared,train_data_labels)
# print(model.feature_importances_) 
# feat_importances = pd.Series(model.feature_importances_, index=['PassengerID','Pclass','Age','SibSp',
#                                                                 'Parch','Fare','Female','Male','C','Q','S'])
# feat_importances.nlargest(12).plot(kind='barh')
# plt.show()


# from sklearn.model_selection import train_test_split

# train_set,val_set,train_set_label,val_set_label = train_test_split(data_prepared,train_data_labels, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=800,min_samples_split=10,
                                    min_samples_leaf=1,max_features='sqrt',
                                    max_depth = 30, criterion = 'entropy',
                                    bootstrap = False,random_state=42)
forest_clf.fit(data_prepared,train_data_labels)

from sklearn.model_selection import cross_val_score
cross_val_score(forest_clf, data_prepared, train_data_labels, cv=3, scoring="accuracy")

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(forest_clf, data_prepared, train_data_labels, cv=3)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(train_data_labels,y_train_pred)


from sklearn.metrics import precision_score, recall_score
precision_score(train_data_labels, y_train_pred)
recall_score(train_data_labels, y_train_pred)

from sklearn.metrics import f1_score
f1_score(train_data_labels, y_train_pred)

# y_hat = forest_clf.predict(val_set)
# cm2 = confusion_matrix(val_set_label,y_hat)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data_prepared,train_data_labels, test_size=0.2, random_state=42)


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
# Number of features to consider at every split
criterion = ['gini','entropy']
max_features = ['auto', 'sqrt',0.2,0.4,1]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2,3, 5, 10,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,5,10,15,20]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)



# Use the random grid to search for best hyperparameters
# First create the base model to tune
forest = RandomForestClassifier(random_state = 42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
forest_random = RandomizedSearchCV(estimator = forest, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# Fit the random search model
forest_random.fit(x_train, y_train)


forest_random.best_params_



from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=800,min_samples_split=10,
                                    min_samples_leaf=1,max_features='sqrt',
                                    max_depth = 30, criterion = 'entropy',
                                    bootstrap = False,random_state=42)
forest_clf.fit(x_train,y_train)

cross_val_score(forest_clf, x_train, y_train, cv=3, scoring="accuracy")


y_hat = forest_clf.predict(x_test)
cm2 = confusion_matrix(y_test,y_hat)





test_data = pd.read_csv('test.csv')
test_data.info()
data_prepared_test = full_pipeline.fit_transform(test_data)


y_pred = forest_clf.predict(data_prepared_test)
Id = test_data['PassengerId'].values

a = np.array([Id,y_pred]).transpose()

df_submission = pd.DataFrame(a,columns = ['PassengerID','Survived'] )

df_submission.to_csv(r'C:\Users\parva\Desktop\Kaggle\Titanic\titanic\submission3.csv',index = False)
