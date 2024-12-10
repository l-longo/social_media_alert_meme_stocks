import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import scipy.stats
from textblob import TextBlob
import statsmodels.formula.api as smf
import statsmodels.api as sm


L1 = 100

path = 'your_path'

amc = pd.read_excel(path + 'amcpval.xlsx')
gme = pd.read_excel(path + 'gmepval.xlsx')
nok = pd.read_excel(path + 'nokpval.xlsx')
tsla = pd.read_excel(path + 'tslapval.xlsx')
bb = pd.read_excel(path + 'bbpval.xlsx')
nvda = pd.read_excel(path + 'nvdapval.xlsx')
aapl = pd.read_excel(path + 'aaplpval.xlsx')
amzn = pd.read_excel(path + 'amznpval.xlsx')
pep = pd.read_excel(path + 'peppval.xlsx')
intc = pd.read_excel(path + 'intcpval.xlsx')


financial = pd.concat([amc.iloc[L1:], gme.iloc[L1:], nok.iloc[L1:], tsla.iloc[L1:],intc.iloc[L1:],
                       bb.iloc[L1:],nvda.iloc[L1:],aapl.iloc[L1:],amzn.iloc[L1:],pep.iloc[L1:]], axis = 0)
financial = financial.reset_index()

financial['rank_k'] = financial['abn_return'][L1:].rank()/(1+L1)
financial['S2_rank'] = np.zeros(financial.shape[0])

financial['S2_rank'] = (1/financial.shape[0]) * np.sum((financial['rank_k'][L1:] - 0.5)**2)
financial['t_rank'] = (financial['rank_k']-0.5)/np.sqrt(financial['S2_rank'])
financial['p_val_rank'] = scipy.stats.norm.sf(abs(financial['t_rank']))
