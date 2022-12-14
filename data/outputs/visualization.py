import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt

file_name = 'vonM_Famous.csv'

df = pd.read_csv(file_name,sep=',',index_col = 0)
index = [x for x in df.index]

plt.errorbar(index,df['train'],yerr=df['train_std']/sqrt(15),label='Train')
plt.errorbar(index,df['LOO'],yerr=df['LOO_std']/sqrt(15),label='Validation')
plt.ylabel('Loglikelihood')
plt.xlabel('Clusters')
plt.legend(); plt.grid(); plt.show()

