# Force reload
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
    return fig
    
def plot_bar(df:pd.DataFrame,col:str, facet_col:str = None):
    if facet_col:
        return px.bar(df,x= col,facet_col = facet_col)
    else:
        return px.bar(df,x = col)

def plot_correlation_matrix(df):
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True)
    return fig

def plot_team_performance(df, team_name):
    fig = px.bar(df, x='year_e', y='team_goals_scored', title=f'Team Performance for {team_name}')
    return fig

def plot_player_comparison(df, players, categories, category_labels=None):
    fig = go.Figure()

    if category_labels is None:
        category_labels = categories

    for player in players:
        player_data = df[df['player_name'] == player]
        if not player_data.empty:
            values = player_data[categories].mean().values.flatten().tolist()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=category_labels,
                fill='toself',
                name=player
            ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 100]
        )),
      showlegend=True
    )

    return fig

def plot_player_stats_bar_chart(df, players, categories, category_labels=None):
    player_data = df[df['player_name'].isin(players)]

    # Aggregate stats over selected seasons
    agg_dict = {cat: 'mean' for cat in categories}
    agg_data = player_data.groupby('player_name').agg(agg_dict).reset_index()
            
    bar_data_melted = agg_data.melt(id_vars=['player_name'], value_vars=categories, var_name='stat', value_name='value')

    if category_labels:
        label_map = dict(zip(categories, category_labels))
        bar_data_melted['stat'] = bar_data_melted['stat'].map(label_map)
    
    fig = px.bar(bar_data_melted, x='value', y='stat', color='player_name', barmode='group', orientation='h', height=1500)
    
    return fig

# validation metrics

def plot_confusion_matrix(cm):
    fig = ff.create_annotated_heatmap(cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'])
    fig.update_layout(title_text='Confusion Matrix')
    return fig

def plot_roc(fpr,tpr,threshold_roc,roc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_score:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash')))
    fig.update_layout(title_text='Receiver Operating Characteristic (ROC) Curve')
    return fig

def plot_precision_recall(recall_curve,precision_curve,pr_auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines', name=f'Precision-Recall curve (area = {pr_auc_score:.2f})'))
    fig.update_layout(title_text='Precision-Recall Curve')
    return fig

def plot_feature_importance(top_25_features):
    fig = px.bar(top_25_features, x='feature_importance', y='feature_names', orientation='h', title='Top 25 Feature Importances')
    return fig

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

def plot_continuous_trend(dat,x,y):
    # Convert 'month_e' (Period) to string to make it compatible with Plotly
    dat[f'{x}_str'] = dat[x].astype(str)

    # Compute average games_rating and standard deviation grouped by month_e_str and team
    agg_data = dat.groupby([f'{x}_str', 'team','year_e']).agg(
        mean_stat=(y, 'mean'),
        std_stat=(y, 'std')
    ).reset_index()
    agg_data['year_e'] = agg_data['year_e'].astype('str')

    # Create the scatter plot with error cloud faceted by team
    fig = px.scatter(
        agg_data,
        x=f'{x}_str',
        y='mean_stat',
        error_y='std_stat',
        color = 'year_e',
        facet_col='team',
        labels={f'{x}_str': x, 'mean_stat': f"mean_{y}"},
        title=f'Average {y} by Month (faceted by Team)',
        height=600
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=f"mean_{y}",
        title_x=0.5
    )

    fig.show()

    return fig