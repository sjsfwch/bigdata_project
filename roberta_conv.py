import tokenizers
import transformers
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch import optim
import argparse
import numpy as np 
import pandas as pd 
import os
import json

MAX_LEN = 96
train_file = 'data/train_fold.csv'
test_file = 'data/test.csv'
sub_file = 'data/sample_submission.csv'
roberta_path = 'roberta-base/'

""" post process
"""
def get_selected_text(text, offsets, start_pos, end_pos, sentiment, length):
    if len(text.split()) <= 3:
        return text
    if start_pos > end_pos:
        end_pos = start_pos
    start_off = offsets[start_pos]
    end_off = offsets[end_pos]
    if start_pos >= length + 1:
        return ''
    if end_pos >= length + 1:
        return text[start_off[0]:]
    return text[start_off[0]: end_off[1]]

""" jaccard
"""
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def preprocess(MAX_LEN, text, sentiment, selected_text=None):
    text1 = ' '+' '.join(text.split())
    enc = tokenizer.encode(text1)
    offsets = enc.offsets
    s_tok = sentiment_d[sentiment]
    
    ids = np.ones((MAX_LEN), dtype=np.int32)
    masks = np.zeros((MAX_LEN), dtype=np.int32)
    types = np.zeros((MAX_LEN), dtype=np.int32)
    start_tokens = np.zeros((MAX_LEN),dtype=np.int32)
    end_tokens = np.zeros((MAX_LEN),dtype=np.int32)
    start_pos = 0
    end_pos = 0
    
    ids[:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    masks[:len(enc.ids)+5] = 1
    offsets = [(0, 0)] + offsets + [(0, 0)] * (MAX_LEN - len(offsets) - 1)
    assert(len(offsets) == MAX_LEN)
    
    if not(selected_text is None):
        text2 = ' '.join(selected_text.split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx:idx+len(text2)]=1
        if text1[idx-1]==' ':
            chars[idx-1] = 1
        toks = []
        for i,(a,b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm>0:
                toks.append(i) 
        if len(toks)>0:
            start_tokens[toks[0]] = 1
            end_tokens[toks[-1]] = 1
        start_pos = toks[0]
        end_pos = toks[-1]
        st = selected_text
    else:
        st = ''
    
    return {
        'ori': text1, 
        'label': sentiment,
        'ori_s': st,
        'offsets': offsets,
        'ids': ids,
        'masks': masks,
        'types': types,
        'start_tokens': start_tokens,
        'end_tokens': end_tokens,
        'start_pos': start_pos,
        'end_pos': end_pos,
        'length': len(enc.ids)
    }

class TweetSet(Dataset):
    def __init__(self, df, MAX_LEN, test=False):
        self.datas = []
        self.ct = df.shape[0]
        self.test = test
        for k in range(self.ct):
            if not test:
                self.datas.append(preprocess(MAX_LEN, df.loc[k, 'text'], df.loc[k, 'sentiment'], df.loc[k, 'selected_text']))
            else:
                self.datas.append(preprocess(MAX_LEN, df.loc[k, 'text'], df.loc[k, 'sentiment']))
    
    def __len__(self):
        return self.ct
    
    def __getitem__(self, x):
        d = self.datas[x]
        result = {}
        tensor_keys = ['ids', 'masks', 'types', 'start_tokens', 'end_tokens', 'start_pos', 'end_pos', 'offsets', 'length']
        for k in d:
            if k in tensor_keys:
                result[k] = torch.tensor(d[k], dtype=torch.long)
            else:
                result[k] = d[k]
        return result

class LinearModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(LinearModel, self).__init__(conf)
        roberta_path = 'roberta-base/'
        
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=conf)
        self.dropout = nn.Dropout(0.1)
        
        self.classifier= nn.Linear(conf.hidden_size * 2, 2)
        nn.init.normal_(self.classifier.weight, std=0.02)
        self.af2 = nn.Softmax(dim=1)
    
    def forward(self, input_ids, input_mask, token_types):
        _, _, out_hiddens = self.roberta(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_types,
        )
        # (bs, MAX_LEN, hidden_size)
        
        x = torch.cat((out_hiddens[-1], out_hiddens[-2]), dim=-1)
        x = self.dropout(x)
        # (bs, MAX_LEN, hidden_size * 2)
        
        x = self.classifier(x)
        x_start, x_end = x.split(1, dim=-1)
        x_start = x_start.squeeze(-1)
        x_end = x_end.squeeze(-1)
        return self.af2(x_start), self.af2(x_end)
        # (bs, MAX_LEN) * 2

""" BCELoss
"""
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        roberta_path = 'roberta-base/'
        self.hs = conf.hidden_size
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=conf)
        self.dropout = nn.Dropout(0.1)
        
        self.cv1 = nn.Conv1d(self.hs * 2, 256, 2, padding=1)
        self.bn = nn.BatchNorm1d(256)
        self.af = nn.LeakyReLU()
        self.cv2 = nn.Conv1d(256, 128, 2) 
        self.af2 = nn.Softmax(dim=1)

        self.classifier= nn.Linear(128, 2)
        nn.init.normal_(self.classifier.weight, std=0.02)
    
    def forward(self, input_ids, input_mask, token_types):
        _, _, out_hiddens = self.roberta(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_types,
        )
        # (bs, MAX_LEN, hidden_size)
        
        x = torch.cat((out_hiddens[-1], out_hiddens[-2]), dim=-1)
        x = self.dropout(x)
        # (bs, MAX_LEN, hidden_size * 2)
        
        x = x.permute([0, 2, 1])
        # (bs, hidden_size * 2, MAX_LEN)
        
        x = self.cv1(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.cv2(x)
        # (bs, 128, MAX_LEN)
        x = x.permute([0, 2, 1])
        x = self.dropout(x)

        x = self.classifier(x)
        x_start, x_end = x.split(1, dim=-1)
        x_start = x_start.squeeze(-1)
        x_end = x_end.squeeze(-1)
        return self.af2(x_start), self.af2(x_end)
        # (bs, MAX_LEN) * 2 
        # after softmax
        
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.f = nn.BCELoss()
    def forward(self, start, end, start_gt, end_gt):
        """ (bs, MAX_LEN), (bs, MAX_LEN)
        """
        return self.f(start, start_gt.float()) + self.f(end, end_gt.float())

class Loss_nll(nn.Module):
    def __init__(self):
        super(Loss_nll, self).__init__()
        self.f = nn.NLLLoss()
    def forward(self, start, end, start_gt, end_gt):
        """ (bs, MAX_LEN), (bs)
        """
        return self.f(torch.log(start), start_gt) + self.f(torch.log(end), end_gt)


def get_parser():
    parser = argparse.ArgumentParser(description="Roberta based Model")
    parser.add_argument('--bs', default='8')
    parser.add_argument('--mp', default='modelvt')
    parser.add_argument('--md', default='linear')
    parser.add_argument('--ls', default='bce')
    return parser

if __name__ == '__main__':
    print('training process: Roberta + Conv')
    args = get_parser().parse_args()
    batch_size = int(args.bs)
    model_path = os.path.join('models', args.mp)
    model_type = args.md
    loss_type = args.ls
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print('batch_size: %s' % batch_size)
    print('model_path: %s' % model_path)
    print('model_type: %s' % model_type)
    print('loss_type : %s' % loss_type)

    """ model config 
    """

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    sub_df = pd.read_csv(sub_file)
    train_df.dropna(inplace=True)
    train_df = train_df.reset_index(drop=True)
    print(train_df.shape, test_df.shape, sub_df.shape)
    """ load data
    """

    tokenizer = tokenizers.ByteLevelBPETokenizer(
        vocab_file=os.path.join(roberta_path, 'vocab.json'), 
        merges_file=os.path.join(roberta_path, 'merges.txt'), 
        lowercase=True,
        add_prefix_space=True
    )
    roberta_config = transformers.RobertaConfig.from_pretrained(roberta_path)
    roberta_config.output_hidden_states = True
    sentiment_d = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
    """ roberta config
    """

    """ training
    """
    n_splits = 5
    max_epochs = 5
    initial_lr = 3e-5
    is_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if is_gpu else 'cpu')

    train_score_list = []
    valid_score_list = []

    for fold in range(n_splits):
        print('fold: %d' % (fold + 1))
        train_set = TweetSet(train_df[train_df['kfold'] != fold].reset_index(), MAX_LEN)
        valid_set = TweetSet(train_df[train_df['kfold'] == fold].reset_index(), MAX_LEN)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_set, batch_size=batch_size)
        model_id = 'roberta-%s-%d-fold.pt' % (model_type, fold + 1)
        print('data loaded & model creating: %s' % model_id)
        if model_type == 'linear':
            net = LinearModel(roberta_config)
        else:
            net = TweetModel(roberta_config)
        if loss_type == 'bce':
            criterion = Loss()
        else:
            criterion = Loss_nll()
        if is_gpu:
            net.cuda()
            criterion.cuda()
        optimizer = transformers.AdamW(net.parameters(), lr=initial_lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: (0.2 ** e))
        
        train_score_list2 = []
        valid_score_max = -np.inf
        for e in range(1, max_epochs + 1):
            train_loss = 0.0
            train_score = 0.0
            valid_score = 0.0
            
            net.train()
            bar = tqdm(train_loader)
            for tp in bar:
                ids, masks, types = tp['ids'], tp['masks'], tp['types']
                if is_gpu:
                    ids, masks, types = ids.cuda(), masks.cuda(), types.cuda()
                if loss_type == 'bce':
                    start_gt, end_gt = tp['start_tokens'], tp['end_tokens']
                else:
                    start_gt, end_gt = tp['start_pos'], tp['end_pos']
                if is_gpu:
                    start_gt, end_gt = start_gt.cuda(), end_gt.cuda()
                
                start_pt, end_pt = net(ids, masks, types)
                loss = criterion(start_pt, end_pt, start_gt, end_gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                offsets = tp["offsets"].numpy()
                ori = tp['ori']
                ori_s = tp['ori_s']
                labels = tp['label']
                start_pt = start_pt.cpu().detach().numpy()
                end_pt = end_pt.cpu().detach().numpy()
                lens = tp['length'].numpy()

                for j, text1 in enumerate(ori):
                    text2 = get_selected_text(
                        text1, offsets[j], 
                        np.argmax(start_pt[j]), 
                        np.argmax(end_pt[j]), 
                        labels[j], lens[j]
                    )
                    train_score += jaccard(ori_s[j], text2)
                
                train_loss += loss.item() * ids.shape[0] / len(train_set)
                bar.set_postfix(ordered_dict={'train_loss': loss.item()})
            
            net.eval()
            with torch.no_grad():
                v_bar = tqdm(valid_loader)
                for tp in v_bar:
                    ids, masks, types = tp['ids'], tp['masks'], tp['types']
                    if is_gpu:
                        ids, masks, types = ids.cuda(), masks.cuda(), types.cuda()
                    start_pt, end_pt = net(ids, masks, types)
                    
                    offsets = tp["offsets"].numpy()
                    ori = tp['ori']
                    ori_s = tp['ori_s']
                    labels = tp['label']
                    start_pt = start_pt.cpu().detach().numpy()
                    end_pt = end_pt.cpu().detach().numpy()
                    lens = tp['length'].numpy()
                    
                    for j, text1 in enumerate(ori):
                        text2 = get_selected_text(
                            text1, offsets[j], 
                            np.argmax(start_pt[j]), 
                            np.argmax(end_pt[j]), 
                            labels[j], lens[j]
                        )
                        valid_score += jaccard(ori_s[j], text2)
            
            train_score /= len(train_set)
            valid_score /= len(valid_set)
            lr = optimizer.param_groups[0]['lr']
            print('[epoch:%s]: train_score=%s, valid_score=%s, lr=%s' % (
                e, train_score, valid_score, lr
            ))
            scheduler.step()
            
            if valid_score > valid_score_max:
                print('[%s > %s]: model update, saving...' % (valid_score, valid_score_max))
                torch.save(net.state_dict(), os.path.join(model_path, model_id))
                valid_score_max = valid_score
            train_score_list2.append(train_score)

        valid_score_list.append(valid_score_max)
        train_score_list.append(train_score_list2)

    """ output result
    """
    result = {
        'train_score': train_score_list,
        'valid_score': valid_score_list,
        'mean_valid_score': np.mean(valid_score_list)
    }
    with open(os.path.join(model_path, 'result.json'), 'w') as fout:
        json.dump(result, fout)