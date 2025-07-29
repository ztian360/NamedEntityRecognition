import os
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from bert_base.train import conlleval
import codecs

def build_corpus(split, make_vocab=True, data_dir="data"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir, split+".txt"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=False)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=False)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)

        tag2id['<PAD>'] = len(tag2id)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

class MyDataset(Dataset):
    def __init__(self,datas,tags,word_2_index,tag_2_index):
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_2_index
        self.tag_2_index = tag_2_index

    def __getitem__(self,index):
        data = self.datas[index]
        tag  = self.tags[index]

        data_index = [self.word_2_index.get(i,self.word_2_index["<UNK>"]) for i in data]
        tag_index  = [self.tag_2_index[i] for i in tag]

        return data_index,tag_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self,batch_datas):
        global device
        datas = []
        tags = []
        batch_lens = []

        for data,tag in batch_datas:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))
        batch_max_len = max(batch_lens)

        # 填充PAD使数据等长
        datas = [i + [self.word_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in datas]
        tags = [i + [self.tag_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in tags]

        return torch.tensor(datas,dtype=torch.int64,device=device),torch.tensor(tags,dtype=torch.long,device=device)



class Mymodel(nn.Module):
    def __init__(self,corpus_num,embedding_num,hidden_num,class_num,bi=True):
        super().__init__()

        self.embedding = nn.Embedding(corpus_num,embedding_num)
        self.lstm = nn.LSTM(embedding_num,hidden_num,batch_first=True,bidirectional=bi)

        if bi :
            self.classifier = nn.Linear(hidden_num * 2,class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        self.cross_loss = nn.CrossEntropyLoss()



    def forward(self,batch_data,batch_tag=None):
        embedding = self.embedding(batch_data)
        out,_ = self.lstm(embedding)

        pre = self.classifier(out)
        self.pre = torch.argmax(pre, dim=-1).reshape(-1)
        if batch_tag is not None:
            loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),batch_tag.reshape(-1))
            return loss




def test(test):
    global word_2_index,model,index_2_tag,device
    while True:
        #text = input("请输入：")
        text = test
        text_index = [[word_2_index.get(i,word_2_index["<PAD>"]) for i in text]]
        text_index = torch.tensor(text_index,dtype=torch.int64,device=device)
        model.forward(text_index)
        pre = [index_2_tag[i] for i in model.pre]
        #print(pre)
        with open('predict.txt','w') as f:
            i=0
            for i in range(len(pre)):
                f.write(pre[i])
                f.write(' ')
        with open('predict.txt', 'r') as file1:
            data = file1.readlines()
            data = data[0].split()
        i = 0
        with open('entity-predict.txt', 'w', encoding='utf-8') as f:
            for line, i in zip(open('data/test.txt', encoding='utf-8'), range(len(data))):
                # for line in open('data/test.txt', encoding='utf-8'):
                if line[0] != '\n':
                    f.write(line[0:-1] + ' ' + data[i])
                    f.write('\n')
                else:
                    f.write('\n')
        eval_result = conlleval.return_report('entity-predict.txt')
        print(''.join(eval_result))
        # 写结果到文件中
        with codecs.open('predict_score.txt', 'a', encoding='utf-8') as fd:
            fd.write(''.join(eval_result))
        break
        #print([f'{w}_{s}' for w,s in zip(text,pre)])




if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_data,train_tag,word_2_index,tag_2_index = build_corpus("train",make_vocab=True) # make_vocab:是否需要构建word_2_index,tag_2_index
    dev_data,dev_tag = build_corpus("dev",make_vocab=False)
    index_2_tag = [i for i in tag_2_index]

    corpus_num = len(word_2_index)
    class_num  = len(tag_2_index)

    epoch = 100
    train_batch_size = 64
    dev_batch_size = 128
    embedding_num = 100
    hidden_num = 100
    bi = True # 双向lstm
    lr = 0.001

    train_dataset = MyDataset(train_data,train_tag,word_2_index,tag_2_index)
    train_dataloader = DataLoader(train_dataset,train_batch_size,shuffle=False,collate_fn=train_dataset.pro_batch_data) # 自己拼接数据，不使用自带的函数

    dev_dataset = MyDataset(dev_data, dev_tag, word_2_index, tag_2_index)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False,collate_fn=dev_dataset.pro_batch_data)

    model = Mymodel(corpus_num,embedding_num,hidden_num,class_num,bi)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    model = model.to(device)

    for e in range(epoch):
        model.train()
        for batch_data,batch_tag in train_dataloader:
            train_loss = model.forward(batch_data,batch_tag)
            train_loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        all_pre = []
        all_tag = []
        for dev_batch_data,dev_batch_tag in dev_dataloader:
            dev_loss = model.forward(dev_batch_data,dev_batch_tag)
            all_pre.extend(model.pre.detach().cpu().numpy().tolist())
            all_tag.extend(dev_batch_tag.detach().cpu().numpy().reshape(-1).tolist())
        # score = f1_score(all_tag,all_pre,)
        print(e,f"dev_loss:{dev_loss:.3f},train_loss:{train_loss:.3f}")
    with open("data/test.txt", "r", encoding='utf-8') as f:
        data = f.readlines()
    i = 0
    a = ''
    for i in range(len(data)):
        a = a + data[i][0]
    test(a)
