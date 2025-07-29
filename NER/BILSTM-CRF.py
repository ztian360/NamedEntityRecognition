import os
from itertools import zip_longest
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
# from seqeval.metrics import f1_score
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
                word_lists.append(word_list+["<END>"])

                tag_lists.append(tag_list+["<END>"])

                word_list = []
                tag_list = []

    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=True)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=True)

    # 如果make_vocab为True，还需要返回word2id和tag_2_index
    if make_vocab:
        word2id = build_map(word_lists)
        tag_2_index = build_map(tag_lists)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)
        word2id["<START>"] = len(word2id)
        # word2id["<END>"]   = len(word2id)

        tag_2_index['<PAD>'] = len(tag_2_index)
        tag_2_index["<START>"] = len(tag_2_index)
        # tag_2_index["<END>"] = len(tag_2_index)
        return word_lists, tag_lists, word2id, tag_2_index
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

        datas = [i + [self.word_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in datas]
        tags = [i + [self.tag_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in tags]

        return torch.tensor(datas,dtype=torch.int64,device=device),torch.tensor(tags,dtype=torch.long,device=device),batch_lens


class Mymodel(nn.Module):
    def __init__(self,corpus_num,embedding_num,hidden_num,class_num,bi=True):
        super().__init__()

        self.embedding = nn.Embedding(corpus_num,embedding_num)
        self.lstm = nn.LSTM(embedding_num,hidden_num,batch_first=True,bidirectional=bi)

        if bi :
            self.classifier = nn.Linear(hidden_num * 2,class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        self.transition = nn.Parameter(torch.ones(class_num, class_num) * 1 / class_num)

        self.loss_fun = self.cal_lstm_crf_loss

    def cal_lstm_crf_loss(self,crf_scores, targets):
        """计算双向LSTM-CRF模型的损失
        该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
        """
        global tag_2_index
        pad_id = tag_2_index.get('<PAD>')
        start_id = tag_2_index.get('<START>')
        end_id = tag_2_index.get('<END>')

        device = crf_scores.device


        batch_size, max_len = targets.size()
        target_size = len(tag_2_index)


        mask = (targets != pad_id)
        lengths = mask.sum(dim=1)
        targets = self.indexed(targets, target_size, start_id)


        targets = targets.masked_select(mask)  # [real_L]

        flatten_scores = crf_scores.masked_select(mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)).view(-1,target_size * target_size).contiguous()

        golden_scores = flatten_scores.gather(dim=1, index=targets.unsqueeze(1)).sum()

        scores_upto_t = torch.zeros(batch_size, target_size).to(device)
        for t in range(max_len):
            batch_size_t = (lengths > t).sum().item()
            if t == 0:
                scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,t, start_id, :]
            else:

                scores_upto_t[:batch_size_t] = torch.logsumexp(
                    crf_scores[:batch_size_t, t, :, :] +
                    scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        all_path_scores = scores_upto_t[:, end_id].sum()

        loss = (all_path_scores - golden_scores) / batch_size
        return loss

    def indexed(self,targets, tagset_size, start_id):
        """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
        batch_size, max_len = targets.size()
        for col in range(max_len - 1, 0, -1):
            targets[:, col] += (targets[:, col - 1] * tagset_size)
        targets[:, 0] += (start_id * tagset_size)
        return targets

    def forward(self,batch_data,batch_tag=None):
        embedding = self.embedding(batch_data)
        out,_ = self.lstm(embedding)

        emission = self.classifier(out)
        batch_size, max_len, out_size = emission.size()

        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition

        if batch_tag is not None:
            loss = self.cal_lstm_crf_loss(crf_scores,batch_tag)
            return loss
        else:
            return crf_scores

    def test(self, test_sents_tensor, lengths):
        """使用维特比算法进行解码"""
        global tag_2_index
        start_id = tag_2_index['<START>']
        end_id = tag_2_index['<END>']
        pad = tag_2_index['<PAD>']
        tagset_size = len(tag_2_index)

        crf_scores = self.forward(test_sents_tensor)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids.reshape(-1)

def test(test):
    global word_2_index,model,index_2_tag,device
    while True:
        # text = input("请输入：")
        text = test
        text_index = [[word_2_index.get(i,word_2_index["<UNK>"]) for i in text] + [word_2_index["<END>"]]]

        text_index = torch.tensor(text_index,dtype=torch.int64,device=device)
        pre = model.test(text_index,[len(text)+1])
        pre = [index_2_tag[i] for i in pre]
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
        # print([f'{w}_{s}' for w,s in zip(text,pre)])




if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_data,train_tag,word_2_index,tag_2_index = build_corpus("train",make_vocab=True)
    dev_data,dev_tag = build_corpus("dev",make_vocab=False)
    index_2_tag = [i for i in tag_2_index]

    corpus_num = len(word_2_index)
    class_num  = len(tag_2_index)

    epoch = 100
    train_batch_size = 64
    dev_batch_size = 128
    embedding_num = 100
    hidden_num = 100
    bi = True
    lr = 0.001

    train_dataset = MyDataset(train_data,train_tag,word_2_index,tag_2_index)
    train_dataloader = DataLoader(train_dataset,train_batch_size,shuffle=False,collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_data, dev_tag, word_2_index, tag_2_index)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False,collate_fn=dev_dataset.pro_batch_data)

    model = Mymodel(corpus_num,embedding_num,hidden_num,class_num,bi)
    opt = torch.optim.AdamW(model.parameters(),lr = lr)
    model = model.to(device)

    for e in range(epoch):
        model.train()
        for batch_data,batch_tag,batch_len in train_dataloader:
            train_loss = model.forward(batch_data,batch_tag)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
            # print(f"train_loss:{train_loss:.3f}")
        model.eval()
        all_pre = []
        all_tag = []
        for dev_batch_data,dev_batch_tag,batch_len in dev_dataloader:
            pre_tag = model.test(dev_batch_data,batch_len)
            all_pre.extend(pre_tag.detach().cpu().numpy().tolist())
            all_tag.extend(dev_batch_tag[:,:-1].detach().cpu().numpy().reshape(-1).tolist())
        # score = f1_score(all_tag,all_pre,average="micro")
        print(e,f"train_loss:{train_loss:.3f}")
    with open("data/test.txt", "r", encoding='utf-8') as f:
        data = f.readlines()
    i = 0
    a = ''
    for i in range(len(data)):
        a = a + data[i][0]
    test(a)

