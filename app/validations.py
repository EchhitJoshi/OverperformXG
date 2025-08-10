from visualizations import *

import pandas as pd
from datetime import datetime
import os


from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay
)


def continuous_evaluations(actual,pred,type = "test"):
    """
    actual: true target
    pred: predicted target
    Check usual continuous target evaluations
    """
    print(f"R-squared(explained variance from the model compared against an average moded): {r2_score(actual,pred)}")
    if r2_score(actual,pred) < 0:
        print("Please check model fit")

    print(f"MAE(Average of absolute Residual): {mean_absolute_error(actual,pred)}")
    print(f"MSE(Average of Squared Residual): {mean_squared_error(actual,pred)}")
    print(f"RMSE(Root Average of Squared Residual): {np.sqrt(mean_squared_error(actual,pred))}")



def discrete_evaluations(actual,pred,pred_proba=None,type = "test",classification_type = None,model_path = ""):
    """
    actual: true target
    pred: predicted target
    pred_proba: predicted probabilities from the model
    path: path to the submodel folder
    Check usual discrete target evaluations
    """
    
    if classification_type != 'Multiclass':
        print(f"Precision: {precision_score(actual, pred,average='weighted')}")
        print(f"Recall: {recall_score(actual, pred,average='weighted')}")
        print(f"F1: {f1_score(actual, pred,average='weighted')}")
        roc_score = roc_auc_score(actual, pred_proba,average='weighted')
        print(f"ROC AUC Score: {roc_score}")
    else:
        print(f"Precision: {precision_score(actual, pred)}")
        print(f"Recall: {recall_score(actual, pred)}")
        print(f"F1: {f1_score(actual, pred)}")
        roc_score = roc_auc_score(actual, pred_proba)
        print(f"ROC AUC Score: {roc_score}")

    if not os.path.exists(model_path + "/metrics"):
        print("creating directory in ", model_path + "/metrics")
        os.mkdir(model_path + "/metrics")
    
    # Model results
    final_result = pd.DataFrame({
        'phase' : [type],
        'datetime': [datetime.now()],
        'precision' : [precision_score(actual, pred)],
        'recall' : [recall_score(actual, pred)],
        'f1' : [f1_score(actual, pred)],
        'auc_score' : [roc_score]

    })

    # Check if the file exists
    csv_file_path = model_path + '/metrics/model_metrics.csv'
    if not os.path.exists(csv_file_path):
        # If the file does not exist, write the header and the new data
        # header=True ensures column names are written
        # index=False prevents writing the DataFrame index as a column
        final_result.to_csv(csv_file_path, mode='a', header=True, index=False)
    else:
        # If the file exists, append the new data without the header
        final_result.to_csv(csv_file_path, mode='a', header=False, index=False)

    # Plots:
    # Confusion Matrix
    if 'test' in type:
        cm = confusion_matrix(actual,pred)

        outcomes = []
        for true, pred in zip(actual, pred):
            if true == 1 and pred == 1:
                outcomes.append('TP')
            elif true == 0 and pred == 0:
                outcomes.append('TN')
            elif true == 0 and pred == 1:
                outcomes.append('FP')
            elif true == 1 and pred == 0:
                outcomes.append('FN')

        # Combine into a DataFrame for easy inspection
        outcomes_df = pd.DataFrame({'actual_class': actual, 'predicted_class': pred, 'predicted_probability':pred_proba,'outcome': outcomes})

        print(outcomes_df.head())
        plot_confusion_matrix(cm)

        # ROC Curve
        fpr, tpr, thresholds_roc = roc_curve(actual, pred_proba)
        plot_roc(fpr,tpr,thresholds_roc,roc_score)


        # Precision Recall Curve
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(actual, pred_proba)
        pr_auc = auc(recall_curve, precision_curve) # Calculate the area under the PR curve
        plot_precision_recall(recall_curve,precision_curve,pr_auc)

        plot_prediction_density(outcomes_df)
    



def check_feature_importance(algorithm_name,trained_model,X_train):
    if algorithm_name.lower() == 'catboost':
        feature_importance = trained_model.get_feature_importance()
        feature_names = X_train.columns
        importance_df = pd.DataFrame({'feature_importance': feature_importance, 'feature_names': feature_names})
        importance_df = importance_df.sort_values(by='feature_importance', ascending=False)
        top_25_features = importance_df.head(25)
    
    plot_feature_importance(algorithm_name,top_25_features)

    return top_25_features



# Probability threshold tuning:
def tune_prob_threshold(actual,pred_proba):
    
    thresholds = np.linspace(0.0,0.1,101)

    threshold = {}

    # For F1
    f1_scores = [f1_score(actual,pred_proba >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    threshold['f1'] = best_threshold
    print(f"Threshold maximizing F1 score: {best_threshold}")
    

    # Youden's J Statistic
    fpr, tpr, thres = roc_curve(actual,pred_proba)
    youden_index = tpr - fpr
    threshold['tpr_fpr'] = thres[np.argmax(youden_index)]
    print(f"Threshold maximizing tpr-fpr: {threshold['tpr_fpr']}")

    return threshold


    