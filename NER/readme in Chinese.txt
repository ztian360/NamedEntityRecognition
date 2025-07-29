重点关注以下代码和文件夹：
BILSTM.py  :双向LSTM的代码
BILSTM-CRF.py  双向LSTM+CRF的代码
datapro.py  ：数据集处理
文件夹BERT-BiLSTM-CRF-NER-master

大体步骤：
1、使用datapro.py对数据集进行处理，原始数据在'People Daily 2014NER'文件夹中，处理好的完整数据集在'people_all_data'文件夹中。
2、用BILSTM.py代码举例，代码文件中包含了读取数据，训练以及测试的代码，在if __name__ == "__main__":处可以修改一些训练参数。
3、最终的结果会保存在后缀为'OUTPUT'的文件夹中。
4、对于BERT-BiLSTM-CRF-NER-master文件，其中包含了介绍文件readme.txt以及需要的环境配置requirement.txt。对于BILSTM和BILSTM-CRF，用到的环境都是python3.7，pytorch1.8.1。