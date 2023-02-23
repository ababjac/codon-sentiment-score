import pandas as pd
import numpy as np
import helpers
import graph


df = pd.read_csv('data/ecoli_complete.csv', index_col=0)
df = helpers.clean(df, 'mRNA_cleaned')
mymethod_vals = np.fromfile('data/testR.csv', sep=',')
mymethod_vals = helpers.norm(mymethod_vals)
df['NLP_CUB'] = mymethod_vals
df['Log Abundance'] = np.log(df['abundance'])

df.rename(columns = {'NLP_CUB':'NLP CUB Measure', 'MM_avg':'%MinMax Average', 'abundance':'Abundance', 'log10phi':'Natural Log of Phi', 'disorder_ratio':'Disorder Ratio'}, inplace = True)

graph.make_scatterplot_data(df, 'NLP CUB Measure', '%MinMax Average', 'Log Abundance', 'Our Method vs Average %MinMax by Abundance', 'images/scatterplots/NLP_MMavg_scatter-log.png')
graph.make_scatterplot_data(df, 'NLP CUB Measure', 'Natural Log of Phi', 'Log Abundance', 'Our Method vs Log(Phi) by Abundance', 'images/scatterplots/NLP_logPhi_scatter-log.png')
graph.make_scatterplot_data(df, 'NLP CUB Measure', 'CAI', 'Log Abundance', 'Our Method vs CAI by Abundance', 'images/scatterplots/NLP_CAI_scatter-log.png')
graph.make_scatterplot_data(df, 'NLP CUB Measure', 'CBI', 'Log Abundance', 'Our Method vs CBI by Abundance', 'images/scatterplots/NLP_CBI_scatter-log.png')
graph.make_scatterplot_data(df, 'NLP CUB Measure', 'Nc', 'Log Abundance', 'Our Method vs Nc by Abundance', 'images/scatterplots/NLP_Nc_scatter-log.png')

#graph.make_scatterplot_data(df, 'NLP CUB Measure', 'Log Abundance', 'Disorder Ratio', 'Our Method vs Abundance', 'images/scatterplots/NLP_abundance_scatter.png')
#graph.make_scatterplot_data(df, 'NLP CUB Measure', 'Log Abundance', '', 'Our Method vs Abundance', 'images/scatterplots/NLP_abundance_scatter-nodisorder.png')
