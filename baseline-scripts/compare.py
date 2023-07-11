import pandas as pd
import numpy as np
import helpers
import graph
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/ecoli_complete.csv', index_col=0)
#df = helpers.clean(df, 'mRNA_cleaned')
#mymethod_vals = np.fromfile('data/testR.csv', sep=',')
#mymethod_vals = helpers.norm(mymethod_vals)
#df['NLP_CUB'] = mymethod_vals
#df['Log Abundance'] = np.log(df['abundance'])

#df['sentiment'] = np.where(df['abundance'] > np.median(df['abundance'].values), 1, 0)
low, high = df.abundance.quantile([0.33, 0.67])
high_l = np.where(df['abundance'] > high, 1, 0)
low_l = np.where(df['abundance'] > low, 0, 2)
df['sentiment'] = high_l+low_l

df_train, df_test = train_test_split(df, test_size=0.2, random_state=62835) #save 20% for testing

#y_pred = np.where(df_test['CBI'] > np.median(df_test['CBI'].values), 1, 0)
#low, high = df_test.CAI.quantile([0.33, 0.67])
#high_l = np.where(df_test['CAI'] > high, 1, 0)
#low_l = np.where(df_test['CAI'] > low, 0, 2)
#y_pred = high_l+low_l
y_pred = pd.read_csv('results/)

print(metrics.accuracy_score(df_test['sentiment'], y_pred))
print(metrics.precision_score(df_test['sentiment'], y_pred, average='macro'))
print(metrics.recall_score(df_test['sentiment'], y_pred, average='macro'))



#print('Abundance vs NLP_CUB', spearmanr(df['abundance'], df['NLP_CUB']))
#print('Abundance vs %MinMaxAvg', spearmanr(df['abundance'], df['MM_avg']))
#print('Abundance vs log10phi', spearmanr(df['abundance'], df['log10phi']))
#print('Abundance vs CAI', spearmanr(df['abundance'], df['CAI']))
#print('Abundance vs CBI', spearmanr(df['abundance'], df['CBI']))
#print('Abundance vs Nc', spearmanr(df['abundance'], df['Nc']))

#df.rename(columns = {'NLP_CUB':'NLP CUB Measure', 'MM_avg':'%MinMax Average', 'abundance':'Abundance', 'log10phi':'Natural Log of Phi', 'disorder_ratio':'Disorder Ratio'}, inplace = True)

#graph.make_scatterplot_data(df, 'NLP CUB Measure', '%MinMax Average', 'Log Abundance', 'Our Method vs Average %MinMax by Abundance', 'images/scatterplots/NLP_MMavg_scatter-log.png')
#graph.make_scatterplot_data(df, 'NLP CUB Measure', 'Natural Log of Phi', 'Log Abundance', 'Our Method vs Log(Phi) by Abundance', 'images/scatterplots/NLP_logPhi_scatter-log.png')
#graph.make_scatterplot_data(df, 'NLP CUB Measure', 'CAI', 'Log Abundance', 'Our Method vs CAI by Abundance', 'images/scatterplots/NLP_CAI_scatter-log.png')
#graph.make_scatterplot_data(df, 'NLP CUB Measure', 'CBI', 'Log Abundance', 'Our Method vs CBI by Abundance', 'images/scatterplots/NLP_CBI_scatter-log.png')
#graph.make_scatterplot_data(df, 'NLP CUB Measure', 'Nc', 'Log Abundance', 'Our Method vs Nc by Abundance', 'images/scatterplots/NLP_Nc_scatter-log.png')

#graph.make_scatterplot_data(df, 'NLP CUB Measure', 'Log Abundance', 'Disorder Ratio', 'Our Method vs Abundance', 'images/scatterplots/NLP_abundance_scatter.png')
#graph.make_scatterplot_data(df, 'NLP CUB Measure', 'Log Abundance', '', 'Our Method vs Abundance', 'images/scatterplots/NLP_abundance_scatter-nodisorder.png')
