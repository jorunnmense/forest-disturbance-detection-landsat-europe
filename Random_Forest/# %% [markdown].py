# %% [markdown]
# ### Creating a Random Forest Baseline to Detect Disturbances in Europe
# 

# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_recall_fscore_support, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import os
from osgeo import gdal
import time
import functools
import seaborn as sns

# %% [markdown]
# Load the Data and print the first lines.
# 

# %%
file_path = '/home/ubuntu/work/saved_data/landsat_disturbance_detection/1D_U_Net/Data/Training/forest_dataset_timesync_alltiles_landsatbands_indices3_classes_ed2_calibration.csv'
df = pd.read_csv(file_path, delimiter=',', index_col=False)
print("Original dataframe", df.head())  

# %% [markdown]
# Remove all NaN values and delete all columns except the NDVI index.

# %%
df = df.dropna()

df = df[df['class_level1'] != 'non-treed']
df = df[df['class_level2'] != 'non-stand-replacing disturbance']

columns_to_delete = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2',
                     'BLU_diff', 'GRN_diff', 'RED_diff', 'NIR_diff'
                     , 'NBR', 'TCB', 'TCG', 'TCW', 'Din'  ] 

df = df.drop(columns=columns_to_delete, errors='ignore')
#print(df[df['numerical_id']==1])

id_counts = df['numerical_id'].value_counts()
filtered_counts = id_counts[id_counts == 3]  # Keep only counts that are NOT 34
print(filtered_counts)

# %% [markdown]
# Create the Train and Testdata. Therefore compute the backward differences.

# %%
#df_select_cols = df[['numerical_id', 'NDVI', 'class_level1','year']].sort_values(['numerical_id', 'year']) #.sort_values(['plotid', 'year'])
#df_grouped = (df_select_cols.groupby('numerical_id'))
df_select_cols = df[['numerical_id', 'NDVI', 'class_level1', 'year']]

# Group by 'numerical_id' and sort within each group
df_grouped = df_select_cols.groupby('numerical_id', group_keys=False).apply(lambda x: x.sort_values('year')).reset_index(drop=True)


# Calculate the differences between the current row and the previous row

df_grouped['NDVI_diff'] = df_grouped.groupby('numerical_id')['NDVI'].diff() # backward difference

print(df_grouped.columns)
print(df_grouped[df_grouped['numerical_id'] == 3441][['numerical_id', 'NDVI', 'year', 'NDVI_diff']])

print(type(df_grouped))

print(df_grouped['numerical_id'].value_counts()[df_grouped['numerical_id'].value_counts() == 1].index)

# Filter out numerical_ids that appear only once
df_filtered = df_grouped.groupby('numerical_id').filter(lambda x: len(x) > 1)

# Split the DataFrame based on 'numerical_id', ensuring the split is done by group
X_train, y_test = train_test_split(df_filtered, test_size=0.2, stratify = df_filtered['numerical_id'], random_state=42)

# Print the resulting train and test datasets
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(y_test)}")

X_train = df_filtered[['NDVI', 'NDVI_diff']].values
y_train = (df_filtered[['class_level1']].values == 'disturbance').astype(int)

X_test = df_filtered[['NDVI', 'NDVI_diff']].values
y_test = (df_filtered[['class_level1']].values == 'disturbance').astype(int)

# %% [markdown]
# Train the model and save it. 

# %%
print(X_train.shape)
print(y_train.shape)

# Flatten y_train and y_test to 1D
#y_train = y_train.ravel()
#y_test = y_test.ravel()

from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier(n_estimators=25, max_depth=None, max_features=None, random_state=0)
model.fit(X_train,y_train)

# %% [markdown]
# Use the test data to evaluate predictions.

# %%
preds = model.predict(X_test)
from sklearn.metrics import f1_score

# Compute overall accuracy
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.4f}")

# Generate a detailed classification report (includes precision, recall, F1-score for each class)
report = classification_report(y_test, preds, target_names=["Class 0", "Class 1"])
print(report)

# %% [markdown]
# Confusion Matrix

# %%
# Compute confusion matrix
cm = confusion_matrix(y_test, preds)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No Disturbance", "Disturbance"], yticklabels=["No Disturbance", "Disturbance"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# %% [markdown]
# Plot the False Negatives.

# %%
print(y_test.shape, preds.shape)

# %%
# Find False Negatives
y_test = y_test.squeeze()
false_negatives_idx = np.where((y_test == 1) & (preds == 0))[0]

# Display some false negatives
print(f"Number of False Negatives: {len(false_negatives_idx)}")

# %%

    ndvi_values = features[ndvi_index, :]

    # Dynamic year labels based on the length of the features (i.e., number of years)
    years = np.arange(1, len(ndvi_values) + 1)

    # Start plotting
    plt.figure(figsize=(12, 6))

    # Plotting the NDVI values (just the NDVI feature)
    plt.plot(years, ndvi_values, marker='o', color='green', label='NDVI')

    # Plotting true labels for disturbance (if applicable)
    added_true_label = False
    for i in range(len(true_label)):
        if true_label[i] == 1:
            if not added_true_label:
                plt.scatter(years[i], true_label[i], color='red', s=100, label='True Disturbance', zorder=5)
                added_true_label = True
            else:
                plt.scatter(years[i], true_label[i], color='red', s=100, zorder=5)

    # Plotting predicted labels for disturbance (if applicable)
    added_predicted_label = False
    for i in range(len(predicted_label)):
        if predicted_label[i] == 1:
            if not added_predicted_label:
                plt.scatter(years[i], predicted_label[i], color='blue', s=100, label='Predicted Disturbance', zorder=5)
                added_predicted_label = True
            else:
                plt.scatter(years[i], predicted_label[i], color='blue', s=100, zorder=5)


# %%
def arrays_to_raster(arrays, cols, rows, transform, projection, output_name):
    """Converts list of NumPy arrays to a multi-band raster file, overwriting existing file if it exists."""
    bands = len(arrays)
    driver = gdal.GetDriverByName('GTiff')

    # Check if the output file already exists, and delete it if it does
    if os.path.exists(output_name):
        os.remove(output_name)

    dataset_index = driver.Create(output_name, cols, rows, bands, gdal.GDT_Float32)
    dataset_index.SetGeoTransform(transform)
    dataset_index.SetProjection(projection)
    
    for i, array in enumerate(arrays, start=1):
        band = dataset_index.GetRasterBand(i)
        band.WriteArray(array)
    dataset_index.FlushCache()

    return dataset_index

# %% [markdown]
# Calculate the NDVI index.

# %%
def calculate_indices(dsimage):
    """Calculates various spectral indices from raster bands."""
    # int und 6 Landsat channels
    array3 = dsimage.GetRasterBand(3).ReadAsArray().astype(float)  # Red
    array4 = dsimage.GetRasterBand(4).ReadAsArray().astype(float)  # NIR
    
    # Calculate indices and tasseled cap 
    # TC coefficients from http://dx.doi.org/10.1080/2150704X.2014.915434
    ndvi = (array4 - array3) / (array4 + array3)          

    return {
        "NDVI": ndvi
    }


