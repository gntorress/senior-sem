import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import pickle

# Define local paths for your datasets
dataset_path = './data/meld-dataset'  # Change this path as per your local directory

print('Data source import complete.')

# Function to display files in directory for verification
def list_files_in_directory(path, max_files=5):
    cpt = 0
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            cpt += 1
            print(os.path.join(dirname, filename))
            if cpt > max_files:
                break

list_files_in_directory(dataset_path)

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = getattr(df, 'dataframeName', 'Dataset')
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

# Load sample datasets
def load_dataset(file_path, rows_to_read=50):
    df = pd.read_csv(file_path, delimiter=',', nrows=rows_to_read)
    df.dataframeName = os.path.basename(file_path)
    print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in {df.dataframeName}')
    return df

# Example loading and plotting data
data_files = {
    "dev": os.path.join(dataset_path, "MELD-RAW/MELD.Raw/dev_sent_emo.csv"),
    "test": os.path.join(dataset_path, "MELD-RAW/MELD.Raw/test_sent_emo.csv"),
    "train": os.path.join(dataset_path, "MELD-RAW/MELD.Raw/train/train_sent_emo.csv")
}

for name, path in data_files.items():
    df = load_dataset(path)
    df.head(5)
    plotPerColumnDistribution(df, 10, 5)
    plotCorrelationMatrix(df, 8)
    plotScatterMatrix(df, 15, 10)

# Training and Testing functions
def train_model(model_obj):
    checkpoint = ModelCheckpoint(os.path.join(model_obj.output_file, "models"), monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model = model_obj.get_model()  # Assume `get_model` based on modality
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(model_obj.train_x, model_obj.train_y, epochs=2, batch_size=model_obj.batch_size, 
              validation_data=(model_obj.val_x, model_obj.val_y),
              callbacks=[early_stopping, checkpoint])

def test_model(model_obj):
    model = load_model(model_obj.PATH)
    # Further testing code with predictions, feature extraction, and pickle saving

# Set classification mode and modality
classify = "Emotion"
modality = "text"
output_file = "./"

print(f"Running in mode: {classify} with modality: {modality}")
# Insert additional training and testing logic based on the modality and classify options
