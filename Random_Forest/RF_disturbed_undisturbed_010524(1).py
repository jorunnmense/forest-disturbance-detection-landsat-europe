# RF model for disturbed vs undisturbed binary classification
# aviana
# May 2024
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_recall_fscore_support, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
#from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

'''Tried different options:
    1.  24 predictors: All bands and indices in t0 (12 columns) and t-1 (12 columns) (i.e. absolute values of a target year and diffs with previous year)
    2.  14 predictors: SW1, SW2 and indices in t0 (8 columns) + diffs indices (6 columns)
    2b. 14 predictors: SW1_diffs, SW2_diffs and indices in t0 (8 columns) + diffs indices (6 columns)
    3.  12 predictors: indices t0 (6 columns) and indices t-1 (6 columns)'''
# With option 2b I got: F1-score 0.99 for class "0" (undisturbed) and F1-score 0.76 for class "1" (disturbed)

### Data preparation
df = pd.read_csv('/home/ubuntu/work/saved_data/landsat_disturbance_detection/clean_1D_U_Net/deep_disturbance/Data/Training/clear_forest_dataset_timesync_alltiles_landsatbands_indices3_classes_ed4_calibration.csv', sep=';')
print("Original dataframe", df.head())  
     
# Remove rows containing NaN values
df = df.dropna()
# Filter out rows where class_level1 is equal to "non-treed"
df = df[df['class_level1'] != 'non-treed']
df = df[df['class_level2'] != 'non-stand-replacing disturbance']

# When using option 3 from the meny: i.e. only indices
# Define the list of column names to delete
columns_to_delete = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2',
                     'BLU_diff', 'GRN_diff', 'RED_diff', 'NIR_diff']  # Add more column names as needed

# Delete columns with specified names
df = df.drop(columns=columns_to_delete, errors='ignore')
print("df modified", df.head())

# label treed and non treed to --> 0 undisturbed and disturbance to 1
class_count = df['class_level1'].value_counts()
print("number of samples per class:", class_count) 

class2= []
for row in df['class_level1']:
    if row == 'disturbance':
        class2.append(1)
    elif row == 'treed': 
        class2.append(0)
df['class2'] = class2

# Split the data into predictors (X) and response (y)
X = df.drop(['fid', 'country', 'plotid', 'year', 'class_level1', 'class_level2', 'merge_id', 'coordx', 'coordy', 'uniqueid', 'class2', 'Tile_ID'], axis=1) # otherwise, select all columns except the last one  data.iloc[:, :-1]
y = df['class2'] 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

### 2) Model preparation
# compute class weights
class_weight_dict = {0: 0.5, 1: 70}   # to balance
# Define the parameter grid to search over
param_grid = {
    'n_estimators': np.arange(50, 150, 10),
    'max_depth': [5, 10, 15],
    'min_samples_split': np.arange(2, 10, 2),
    'min_samples_leaf': np.arange(1, 10, 2),
    'max_features': ['sqrt', 'log2', None], 
    'bootstrap': [True, False], 
    'class_weight': ['balanced', 'balanced_subsample', class_weight_dict] 
} 

# Train the random forest classifier with RandomizedSearchCV and oversample the minority
clf = RandomForestClassifier(n_jobs=5) 
# Create an instance of the RandomOverSampler
oversampler = RandomOverSampler()
# Create a pipeline with the oversampler and random forest classifier
pipeline = make_pipeline(oversampler, clf)

# Define the parameter distributions for the random search
param_distributions = {
    'randomforestclassifier__n_estimators': param_grid['n_estimators'],
    'randomforestclassifier__max_depth': param_grid['max_depth'],
    'randomforestclassifier__min_samples_split': param_grid['min_samples_split'],
    'randomforestclassifier__min_samples_leaf': param_grid['min_samples_leaf'],
    'randomforestclassifier__max_features': param_grid['max_features'],
    'randomforestclassifier__bootstrap': param_grid['bootstrap'],
    'randomforestclassifier__class_weight': param_grid['class_weight']
}

random_search = RandomizedSearchCV(pipeline, param_distributions, n_jobs=5, scoring='f1') #, n_jobs=-1, verbose=1, error_score='raise') 
random_search.fit(X_train, y_train)

# Get the best parameters found by Random Search
best_params = random_search.best_estimator_.get_params()
# Filter out unexpected parameters
best_params = {k: v for k, v in best_params.items() if k in random_search.get_params()}
# Create a new Random Forest model with the best parameters
best_rf = RandomForestClassifier(**best_params)
# Train the model
best_rf.fit(X_train, y_train)

# Evaluate on the training data
y_train_pred = random_search.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)
# Evaluate on the testing data
y_test_pred = random_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)
  
# Predict on the test data - to fine tunning probablities if useful
proba = random_search.predict_proba(X_test) #_test)#[:, 1]
# predict
y_pred = random_search.predict(X_test)  
# set a custom threshold
#threshold = 0.4
# classify the samples using the custom threshold if needed (by default RF uses 0.5 threshold)
#y_pred = np.where(proba[:, 1] >= threshold, 1, 0)      


### 3) Evaluate the model's performance on the test sample
acc = accuracy_score(y_test, y_pred) 
print('**************')
print("overall accuracy", acc * 100)
# per class evaluation
test_f1 = f1_score(y_test, y_pred) 
print ('f1 score classification:', test_f1)  

# Define a range of thresholds to try and test performances when variating the probab.
thresholds = np.arange(0.1, 1.0, 0.1)
# Evaluate the performance of the classifier for each threshold
for threshold in thresholds:
    y_pred = (proba[:,1] >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)  #y_test
    precision = precision_score(y_test, y_pred)  # y_test
    recall = recall_score(y_test, y_pred)  # y_test
    roc_auc = roc_auc_score(y_test, proba[:,1])  # y_test
    print(f"Threshold: {threshold:.1f} | F1-score: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | ROC-AUC: {roc_auc:.3f}")

print('**************')
print("Best hyperparameters:", random_search.best_params_)
print('**************')
print("Best OOB score:", random_search.best_score_)
print('**************')
print("best estimator", random_search.best_estimator_)
print('**************')
    
# Get the best estimator from the random search
best_estimator = random_search.best_estimator_
# Get the feature importances of the best estimator
feature_importances = best_estimator.named_steps['randomforestclassifier'].feature_importances_
   
# Calculate precision, recall, and F1 score for each class
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None)  

# Print the precision, recall, and F1 score for each class
for i, class_name in enumerate(np.unique(y_test)):
   print("Class: {}".format(class_name))
   accuracy = accuracy_score(y_test[y_test == class_name], y_pred[y_test == class_name])  
   print("Accuracy class:", accuracy)
   print("Precision: {:.2f}".format(precision[i]))
   print("Recall: {:.2f}".format(recall[i]))
   print("F1 Score: {:.2f}".format(f1_score[i]))
   print("")

# Create a DataFrame with y_pred, y_test, and X_test
result_df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test, 'X_test': X_test.values.tolist()})
# Save the DataFrame to a CSV file
#result_df.to_csv("/xxx/xxxx.csv", index=False)

# CONFUSION MATRIX
label_mapping = {0: "undisturbed", 1: "disturbed"}
cm = confusion_matrix(y_test, y_pred, labels=random_search.classes_) 
print("***raw confusion matrix*****", cm)
# Calculate percentages
cm_percent = (cm / cm.sum(axis=1)[:, np.newaxis]) * 100
# Set the custom labels as tick labels
tick_labels = [label_mapping[label] for label in random_search.classes_]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent,
                             display_labels=tick_labels)
cmap = plt.get_cmap('Blues')
disp.plot(values_format='.2f', cmap=cmap)

# PROD vs USR accuracies
user_acc = cm.diagonal() / cm.sum(axis=1)
print('user acc:', user_acc)
producer_acc = cm.diagonal() / cm.sum(axis=0)
print('producer acc:', producer_acc)

# Compute the AUC of the PR curve using the trapezoidal rule
pr_auc = auc(recall, precision)
print ("pr_auc", pr_auc)
### Compute the false positive rate and true positive rate for different probability thresholds
fpr, tpr, thresholds = roc_curve(y_test, proba[:,1]) 
# Compute the area under the ROC curve (AUC)
roc_auc = roc_auc_score(y_test, proba[:,1])  

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([-0.05, 1.05], [-0.05, 1.05], 'k--', label='Random guess')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate: 1- sensivity')
plt.ylabel('True Positive Rate: sensivity')
plt.title('Receiver Operating Characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()