#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 12:08:31 2018

@author: sjjoo
"""
#%%
import numpy as np
from sklearn import linear_model as lm
import statsmodels.api as sm

import statsmodels.formula.api as smf
import pandas as pd

X = np.column_stack((temp_read,temp_raw, temp_age,temp_meg1, temp_meg2))
y = temp_meg2

reg = lm.LinearRegression()
reg.fit(X,y)


d = pd.DataFrame(X,columns=['read', 'raw', 'age','meg_dot', 'meg_lex'])
result1 = smf.ols('meg_dot~read',d).fit()
result2 = smf.ols('meg_dot~age',d).fit()

result3 = smf.ols('meg_dot~read+age',d).fit()
print(result1.summary())
print(result2.summary())
print(result3.summary())

result4 = smf.ols('meg_dot~meg_lex',d).fit()

print(result4.summary())

result5 = smf.ols('meg_dot~raw+age',d).fit()

print(result5.summary())