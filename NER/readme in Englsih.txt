Focus on the following code files and folders:

BILSTM.py: Code for Bi-directional LSTM
BILSTM-CRF.py: Code for Bi-directional LSTM + CRF
datapro.py: Data processing script
Folder: BERT-BiLSTM-CRF-NER-master

General Steps:

1. Use datapro.py to process the dataset. The raw data is in the 'People Daily 2014NER' folder, and the processed dataset is stored in the 'people_all_data' folder.
2. Take BILSTM.py as an example. This script includes code for reading data, training, and testing. You can modify some training parameters in the if __name__ == "__main__": section.
3. The final results will be saved in a folder with the suffix 'OUTPUT'.
4. The BERT-BiLSTM-CRF-NER-master folder contains an introduction file readme.txt and the required environment configuration requirement.txt.
For BILSTM and BILSTM-CRF, the required environment is Python 3.7 and PyTorch 1.8.1.