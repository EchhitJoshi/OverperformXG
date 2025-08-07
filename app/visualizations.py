import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import gaussian_kde

"""
General functions for plots and facets
Some customizations for dashboarding, functions suffixed with _d
"""


def plot_pie(df:pd.DataFrame,col:str, facet_col:str = None):
    if facet_col:
        fig = px.pie(df,names = col,facet_col = facet_col)
    else:
        fig = px.pie(df,names = col)
    fig.show()
    
def plot_bar(df:pd.DataFrame,col:str, facet_col:str = None):
    if facet_col:
        return px.bar(df,x= col,facet_col = facet_col)
    else:
        return px.bar(df,x = col)
    


# validation metrics

def plot_confusion_matrix(cm):
    cm_plot = ConfusionMatrixDisplay(cm)
    cm_plot.plot()
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc(fpr,tpr,threshold_roc,roc_score):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall(recall_curve,precision_curve,pr_auc_score):
    plt.figure()
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

def plot_feature_importance(top_25_features):
    plt.figure(figsize=(12, 10)) 
    plt.barh(top_25_features['feature_names'], top_25_features['feature_importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Top 25 Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()


def plot_prediction_density(outcomes_df):
    '''
    outcomes_df: df with actual, predicted, predicted_probability, and tp/fp/fn/tp labels
    '''
    kde_dfs = []

    for pred_class,dat in outcomes_df.groupby('outcome'):
        kde = gaussian_kde(dat['predicted_probability'])
        x_vals = np.linspace(dat['predicted_probability'].min() - 1,dat['predicted_probability'].max() + 1,200)
        y_vals = kde(x_vals)
        kde_df = pd.DataFrame({'predicted_probabilities':x_vals,
                               'density': y_vals,
                               'class':pred_class})
        kde_dfs.append(kde_df)
    
    #All kdes
    kde_all = pd.concat(kde_dfs)
    fig = px.line(kde_all, x = 'predicted_probabilities', y = 'density', color = 'class',title = 'class distributions')
    fig.update_layout(xaxis_range = (0.5,1))
    fig.show()

    # fig,axes = plt.subplots(1,1,figsize = (15,8))
    # sns.kdeplot(data= outcomes_df,x = 'predicted_probability',hue = 'outcome',ax =axes)
    # axes.set_xlabel("predicted probabilities")
    # axes.set_ylabel("density")
    # axes.set_title("predicted class distribution")
    # fig.show()