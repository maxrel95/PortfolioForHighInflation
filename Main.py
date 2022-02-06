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

