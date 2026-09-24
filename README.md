## Engineering thesis
This repository is a part of engineering thesis for building a multimodal early-fusion model for emotion recognition using facial expressions and biosignals based on the EMBOA project dataset made together with Adam Sobczuk at Gdansk University of Technology. It works as following pipeline:

-Data Processing
  -Extracts face embeddings from video files (.mp4, .MTS) using MTCNN for face detection and InceptionResnetV1 (FaceNet)
  -Processes and synchronizes physiological biosignals, specifically Electrodermal Activity (EDA), Temperature (TEMP), and Heart Rate (HR)
  -Trims and aligns data vectors across modalities based on timestamps and recording frequencies to create unified input files
  -Cleans and processes BORIS annotation files, supporting two different behavioral labeling methods
-Exploratory Data Anaysis
  -Constructs and inspects the dataset folder hierarchy across the GUT, ITU-YU, and MAAP centers
  -Counts file types and tracks available camera angles per session
  -Evaluates the distribution of emotion labels (Happy, Sad, Scared, Disgusted, Surprised, Angry, and Neutral) across the datasets
  -Identifies missing or zero-value face embeddings and compares them against labeled and unlabeled data rows
-Modeling and Evaluation
  -Utilizes TensorFlow and Keras to train and evaluate two distinct predictive models using balanced datasets
  -Model I: Employs a custom masked categorical cross-entropy loss function for emotion classification tasks.
  -Model II: Employs a custom masked mean squared error loss function
  -Both models leverage Bidirectional LSTM architectures combined with Dropout and Dense layers
  -Evaluates model performance using comprehensive metrics including Accuracy, Precision, Recall, F1-Score, Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and overall similarity.

### VectorCutting.ipynb
Data collection, synchronisation of modalities and preprocessing. Handles the multimodal feature extraction and synchronization process. It extracts facial features from video frames using MTCNN and FaceNet, calculates accurate initial timestamps using video metadata and ffprobe, and aligns the video data with physiological biosignals (EDA, TEMP, HR). Finally, it averages and slices the vectors to match frequencies, exporting the synchronized data into new input and annotation files.

### DataWrangling.ipynb
Cleaning of unusual data points in files for the purpose of further processing. It filters out a comprehensive list of null-value formats and trims trailing empty rows to prepare the CSVs for further processing.

### CountingData.ipynb
Exploratory and descriptive analysis of the preprocessed dataset. Analyzes data integrity by comparing row counts across Method I, Method II, and input files to ensure consistency. It also calculates the percentage distribution of different emotion labels and visualizes the proportion of missing versus detected facial embeddings.

### Models.ipynb
Prepares the data for training by normalizing features, applying one-hot encoding, and generating time-series sequences. It defines the Bidirectional LSTM architectures for both Model I and Model II and executes the model training process.

### Metrics.ipynb
It defines custom masked loss functions (categorical cross-entropy and mean squared error) and evaluates the trained Bidirectional LSTM models. This notebook also generates confusion matrices, visualizes evaluation metrics via heatmaps and bar plots, and exports the final performance tables into LaTeX format.

### ffprobe.exe 
This file is necessary for running of code blocks in VectorCutting.ipynb. 

### Plots
This directory contains created plots for purpose of visual analysis of datasets, and models' results.

### LatexTables
This directory contains Latex tables created for exporting to Overleaf to be included in the thesis.

### AdamSobczukOskarKołoszkoEngineeringThesis.pdf
The thesis in question.
