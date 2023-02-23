import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import seaborn as sns

def plot_confusion_matrix(y_pred, y_actual, title, filename):
    plt.gca().set_aspect('equal')
    cf_matrix = metrics.confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    #print(cf_matrix)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_auc(y_pred, y_actual, title, filename):
    fpr, tpr, thresholds = metrics.roc_curve(y_actual, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def make_scatterplot_data(df, X, Y, color_by, plt_title, path):
    sm_cmap = plt.cm.get_cmap('RdBu')
    if color_by == '':
        df.plot.scatter(X, Y, s=8, title=plt_title)
    else:
        df.plot.scatter(X, Y, c=color_by, s=8, colormap=sm_cmap, title=plt_title)
    plt.savefig(path)
    plt.close()


def make_histogram_data(X, xlabel, plt_title, bins, path):
    plt.hist(X, bins)
    plt.title(plt_title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.savefig(path)
    plt.close()
