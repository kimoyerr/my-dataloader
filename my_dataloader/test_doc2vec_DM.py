#!/usr/bin/env python
"""Creates a Doc2Vec DM model: https://medium.com/@amarbudhiraja/understanding-document-embeddings-of-doc2vec-bfe7237a26da


"""

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

from dataset import SeqDataset, KmerTokenize
from models import DM
from models import NegativeSampling



# Function to convert string arguments to boolean if needed
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Time counter
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def create_dataset(file_name):
    # Create PyTorch Dataset
    fn = os.path.join(os.getcwd(), file_name)

    # Create DataSet and DataLoader
    batch_size = 1
    context_size = 3
    kmer_hypers = {'k': 4, 'overlap': False, 'merge': True}
    ds = SeqDataset(fn, kmer_size=kmer_hypers['k'], context_size=context_size,
                      transform=transforms.Compose([KmerTokenize(kmer_hypers)]))

    return ds

def test_DM_model(SeqData, batch_size=4, shuffle=False, pin_memory=False, num_workers=0, use_gpu=0):

    # Create DataLoader
    print(num_workers)
    loader = DataLoader(SeqData, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                num_workers=num_workers)

    print(torch.cuda.is_available())
    if torch.cuda.is_available() and use_gpu:
        use_gpu = 1
        cuda0 = torch.device('cuda:0')

    else:
        use_gpu = 0
    print(use_gpu)

    # Embedding parameters
    vec_dim = 24
    num_docs = len(SeqData.doc_to_ix)
    num_words = len(SeqData.word_to_ix)
    print(num_docs)

    # Training parameters
    train_batch_size = batch_size # How many target words to use for training in each batch
    n_epochs = args.num_epochs

    # Model initialization
    model = DM(vec_dim, num_docs=num_docs, num_words=num_words)
    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=0.1)
    if use_gpu:
        model.to('cuda')

    # Training

    # Keep track of time
    torch.cuda.synchronize()
    data_time = AverageMeter('Data', ':6.3f')
    data_end = time.time()
    data_end = time.time()

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for epoch_i in range(n_epochs):
            print(epoch_i)
            epoch_start_time = time.time()
            loss = []
            n_batches = 0

            for i, batch in enumerate(loader):
                data_time.update(time.time() - data_end)
                data_end = time.time()

                doc_ids = batch[0]
                target_ids = batch[1]
                context_ids = batch[2]
                target_noise_ids = batch[3]

                # CUDA check
                if use_gpu:
                    doc_ids.to('cuda', non_blocking=True)
                    target_ids.to('cuda', non_blocking=True)
                    context_ids.to('cuda', non_blocking=True)
                    target_noise_ids.to('cuda', non_blocking=True)

                # Do the forward and backward low level batches
                n_train_batches = 0
                train_batch_size = 1
                for j in range(0, doc_ids.shape[0], train_batch_size):
                    train_doc_ids = doc_ids[j:j+train_batch_size,:].squeeze()
                    train_target_ids = target_ids[j:j+train_batch_size,:].squeeze()
                    train_context_ids = context_ids[j:j + train_batch_size,:].squeeze()
                    train_target_noise_ids = target_noise_ids[j:j + train_batch_size, :].squeeze()

                    # Zero-grad is important
                    optimizer.zero_grad()

                    # Model forward pass
                    x = model.forward(
                        train_context_ids,
                        train_doc_ids,
                        train_target_noise_ids)

                    # Negative sampling cost function
                    x_cost = cost_func.forward(x)
                    x_cost.backward()
                    optimizer.step()

                    n_train_batches += 1
                n_batches += 1


            # end of epoch
            epoch_total_time = round(time.time() - epoch_start_time)
            print('Time taken for epoch:')
            print(epoch_total_time)

        if use_gpu==1:
            # Release GPU cache
            torch.cuda.empty_cache()

        # CPU consumption
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


def main(args):
    SeqData = create_dataset(args.file_name)
    test_DM_model(SeqData, args.batch_size, args.shuffle, args.pin_memory, args.num_workers, args.use_gpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Data parameters
    parser.add_argument('--file-name', type=str, default=None, metavar='FN',
                        help='The name of the file to get the sequences from (default: None)')
    parser.add_argument('--num-epochs', type=int, default=1, metavar='NE',
                        help='The batch size to use for dataloader (default: 1)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS',
                        help='The batch size to use for dataloader (default: 4)')
    parser.add_argument('--shuffle', type=str2bool, nargs='?', const=True, default=False,
                        help='Should the sequences be shuffled before batching and serving (default: False')
    parser.add_argument('--pin-memory', type=str2bool, nargs='?', const=True, default=False,
                        help='Should the outputs from dataloader be pinned before transfering to GPU (default: False')
    parser.add_argument('--num-workers', type=int, default=0, metavar='NW',
                        help='The number of workers to use for dataloader (default: 0)')
    parser.add_argument('--use-gpu', type=str2bool, nargs='?', const=True, default=False,
                        help='Should GPU be used (default: False')
    print(parser.parse_args())

    main(parser.parse_args())
