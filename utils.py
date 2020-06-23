import torch
import torch.nn as nn
import numpy as np

def bbox_collate(batch):
    collated = {}
    
    for key in batch[0]:
        collated[key] = [torch.from_numpy(b[key]) for b in batch]
    
    collated['img'] = torch.stack(collated['img'], dim=0).to(torch.float)
    
    return collated

#入力データ(バッチ)から教師データに変換　変換後：[[0,0,1],[0,0,0],...]
def data2target(data, output):
    target = torch.zeros_like(output)
    n,t,v,u = 0, 0, 0, 0
    for i in range(output.shape[0]):
        bbox = data["annot"][i][:,:]
        bbox = bbox[bbox[:,4]!=-1]
        flag = -1
        for k in range(bbox.shape[0]):
            flag = int(bbox[k][4])
            target[i][flag] = 1
            n+=1
        if flag == 0:
            t+=1
        elif flag == 1:
            v+=1
        elif flag == 2:
            u+=1
    target.cuda()
    return target,n,t,v,u

def calc_confusion_matrix(output, target, gt):
    tp = (output * target).sum(axis = 0)
    fp = (output * (1 - target)).sum(axis = 0)
    fn = gt[1:] - tp
    tn = np.all((1 - output) * (1 - target), axis = 1).sum()
    return tp, fp, fn, tn

class InfiniteSampler:
    '''
    与えられたLength内に収まる数値を返すIterator
    '''
    def __init__(self, length, random=True, generator=None):
        self.length = length
        self.random = random
        if random:
            self.generator = torch.Generator() if generator is None else generator
        self.stock = []
        
    def __iter__(self):
        while True:
            yield self.get(1)[0]
    
    def get(self, n):
        while len(self.stock) < n:
            self.extend_stock()
        
        indices = self.stock[:n]
        self.stock = self.stock[n:]
        
        return indices
        
    def extend_stock(self):
        if self.random:
            self.stock += torch.randperm(self.length, generator=self.generator).numpy().tolist()
        else:
            self.stock += list(range(self.length))


class MixedRandomSampler(torch.utils.data.sampler.Sampler):
    '''
    複数のデータセットを一定の比で混ぜながら、指定した長さだけIterationするSampler
    '''
    def __init__(self, datasets, length, ratio=None, generator=None):
        self.catdataset = torch.utils.data.ConcatDataset(datasets)
        self.length = length
        
        self.generator = torch.Generator() if generator is None else generator
        
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        if ratio is None:
            self.ratio = torch.tensor(self.dataset_lengths, dtype=torch.float)
        else:
            self.ratio = torch.tensor(ratio, dtype=torch.float)
            
        self.samplers = [InfiniteSampler(l, generator=self.generator) for l in self.dataset_lengths]
    
    def __iter__(self):
        start_with = torch.cumsum(torch.tensor([0] + self.dataset_lengths), dim=0)
        selected = self.random_choice(self.ratio, self.length)
        
        indices = torch.empty(self.length, dtype=torch.int)
        
        for i in range(len(self.ratio)):
            mask = selected == i
            n_selected = mask.sum().item()
            indices[mask] = torch.tensor(self.samplers[i].get(n_selected), dtype=torch.int) + start_with[i]
        
        indices = indices.numpy().tolist()[0::1]
        
        return iter(indices)
    
    def __len__(self):
        return int(self.length)
    
    def get_concatenated_dataset(self):
        return self.catdataset
    
    def random_choice(self, p, size):
        random = torch.rand(size, generator=self.generator)
        bins = torch.cumsum(p / p.sum(), dim=0)
        choice = torch.zeros(size, dtype=torch.int)

        for i in range(len(p) - 1):
            choice[random > bins[i]] = i + 1

        return choice

def draw_graph(recall, specifity, seed, val, epoch, iteration, it, viz):
    metric = recall + specifity - 1
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([metric[0]]), \
                                win=f'metric{seed}', name='torose', update='append',
                                opts=dict(showlegend=True,title=f"Recall+Specifity-1 val{val}"))
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([metric[1]]), \
                                win=f'metric{seed}', name='vascular', update='append',
                                opts=dict(showlegend=True))
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([metric[2]]), \
                                win=f'metric{seed}', name='ulcer', update='append',
                                opts=dict(showlegend=True))
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([metric.mean()]), \
                                win=f'metric{seed}', name='average', update='append',
                                opts=dict(showlegend=True))

    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([recall[0]]), \
                                win=f'rs0{seed}', name='recall', 
                                update='append',opts=dict(showlegend=True, title=f"Torose{val}"))
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([recall[1]]), \
                                win=f'rs1{seed}', name='recall',
                                update='append', opts=dict(showlegend=True, title=f"Vascular{val}"))
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([recall[2]]), \
                                win=f'rs2{seed}', name='recall', 
                                update='append',opts=dict(showlegend=True, title=f"Ulcer{val}"))
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([specifity[0]]), \
                                win=f'rs0{seed}', name='specifity', 
                                update='append',opts=dict(showlegend=True))
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([specifity[1]]), \
                                win=f'rs1{seed}', name='specifity',
                                update='append', opts=dict(showlegend=True))
    viz.line(X = np.array([it + epoch*iteration]),Y = np.array([specifity[2]]), \
                                win=f'rs2{seed}', name='specifity', 
                                update='append',opts=dict(showlegend=True))