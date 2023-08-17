from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
cm = confusion_matrix(y_test, svc_pred, labels= [0,1,2])
cmd_obj = ConfusionMatrixDisplay(cm)
cmd_obj.plot()
cmd_obj.ax_.set(
                title='Matrica konfuzije',
                xlabel='Predviđene oznake',
                ylabel='Točne oznake',
                )
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

    plt.figure(figsize=figsize)


    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = [0,1,2,"avg"] 

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd, zero_division="warn")).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                cmap='twilight',
                annot=True,
                cbar=True,
                xticklabels=xticks,
                yticklabels=yticks,
                linewidths=.5,
                ax=ax,
                vmin=0.1,
                vmax=1)
plot_classification_report(y_test, svc_pred)

plt.show()