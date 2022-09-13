from sqlite3 import paramstyle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel('Data/data.xlsx', header=0, index_col=0)

df.iloc[:, 0:2].plot()
plt.show()

m, b = np.polyfit(df.iloc[:, 0], df.iloc[:, 2], 1)

df.plot.scatter(x=df.columns[0], y=df.columns[2])
plt.plot(df.iloc[:, 0], m*df.iloc[:, 0] + b, 'r')
plt.xlabel('Inflation')
plt.ylabel('Market')
plt.show()

highInflation = df.iloc[:, 0] > 3
lowInflation = df.iloc[:, 0] <= 1
mediumInflation = ~highInflation & ~lowInflation

returnBySectorHighInflation = df.iloc[highInflation.values, 3:].mean(axis=0)
returnBySectorMediumInflation = df.iloc[mediumInflation.values, 3:].mean(axis=0)
returnBySectorLowInflation = df.iloc[lowInflation.values, 3:].mean(axis=0)

averageRetInRegime = pd.concat([returnBySectorHighInflation,
                                returnBySectorMediumInflation, returnBySectorLowInflation],
                               axis=1)
averageRetInRegime.columns = ['HighInflation', 'MediumInflation', 'LowInflation']

averageRetInRegime.plot.bar()
plt.ylabel('Monthly average return')
plt.show()

highMinusLow = averageRetInRegime['HighInflation'] - averageRetInRegime['LowInflation']

highMinusLow.plot.bar()
plt.ylabel('Monthly Average return')
plt.legend('High - Low')
plt.show()

df_factor = pd.read_excel('Data/AllFactor15.xlsx', header=0, index_col=0)

returnByFactorHighInflation = df_factor.iloc[highInflation.values, :].mean(axis=0)
returnByFactorMediumInflation = df_factor.iloc[mediumInflation.values, :].mean(axis=0)
returnByFactorLowInflation = df_factor.iloc[lowInflation.values, :].mean(axis=0)

averageRetInRegimeFactor = pd.concat([returnByFactorHighInflation,
                                returnByFactorMediumInflation, returnByFactorLowInflation],
                               axis=1)
averageRetInRegimeFactor.columns = ['HighInflation', 'MediumInflation', 'LowInflation']

averageRetInRegimeFactor.plot.bar()
plt.ylabel('Monthly average return')
plt.show()

highMinusLowFactor = averageRetInRegimeFactor['HighInflation'] - averageRetInRegimeFactor['LowInflation']

highMinusLowFactor.plot.bar()
plt.ylabel('Monthly Average return')
plt.legend('High - Low')
plt.show()

mktv = df['Mktv'] 

sorted_mktv = mktv.sort_values()
T = sorted_mktv.__len__()
ecdf = pd.DataFrame([(sorted_mktv <= rt).sum() / T for rt in sorted_mktv.to_list() ])

from scipy.stats import norm

tcdf = pd.DataFrame(norm.cdf(sorted_mktv, loc=mktv.mean(), scale=mktv.std()))

ks = (ecdf - tcdf).abs().max()

plt.figure()
plt.plot(tcdf)
plt.plot(ecdf)

abs_diff = (ecdf - tcdf).abs()
pos = np.where(abs_diff == ks)

critical_value = pd.DataFrame([0.805/np.sqrt(T), 0.886/np.sqrt(T), 1.031/np.sqrt(T)],
index=['10%', '5%', '1%'])

if any( ks > critical_value ):
    print( f'we reject H_0 and conclude about non normality at {critical_value[ks > critical_value].index.values}' )
else:
    print( f'we can reject H_0 at {critical_value[ks > critical_value].index.values} and conclude about nothing' )

# try to fit an ar(1) by ML
x = mktv.iloc[:-1].values
y = mktv.iloc[1:].values

import statsmodels.api as sm
model = sm.OLS(y,x)
results = model.fit()
results.params

starting = [.09698771, 0, 0.1]

def ar_normal( params, x, y ):
    x = x.reshape(( -1, 1 ))
    y = y.reshape(( -1, 1 ))
    nbr_regressor = x.shape[ 1 ]
    eps = y - x*params[ :nbr_regressor ]
    loglik = np.sum( norm.logpdf( eps, loc=params[nbr_regressor ], scale=params[ -1 ] ) )
    return -loglik

from scipy.optimize import minimize
res = minimize(ar_normal, starting, (x,y), method='Nelder-Mead', options={'gtol': 1e-6, 'disp': True})



