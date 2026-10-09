"""Held-out binary evaluation helpers; use only on the designated test set."""
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

def evaluate(y_true, probabilities):
    y_pred=probabilities.argmax(axis=1)
    return {'confusion_matrix':confusion_matrix(y_true,y_pred,labels=[0,1]).tolist(),'classification_report':classification_report(y_true,y_pred,labels=[0,1],target_names=['Normal','Abnormal'],output_dict=True,zero_division=0),'roc_auc':float(roc_auc_score(y_true,probabilities[:,1])),'pr_auc':float(average_precision_score(y_true,probabilities[:,1]))}
