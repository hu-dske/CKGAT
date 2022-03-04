import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='music',
                    help='which dataset to use (music, book, movie, restaurant)')
parser.add_argument('--n_epoch', type=int, default=20, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--n_layer', type=int, default=3, help='depth of layer')  
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--ng', type=float, default=1e-2, help='negative_slope of the LeakyReLU')

parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--user_triple_set_size', type=int, default=8, help='the number of triples in triple set of user')
parser.add_argument('--item_triple_set_size', type=int, default=64, help='the number of triples in triple set of item')
parser.add_argument('--neighbor_size', type=int, default=3, help='the number of neighbor triples')
parser.add_argument('--agg', type=str, default='attention', help='the type of aggregator (sum, pool, concat, attention)')

parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=True, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

args = parser.parse_args()


def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)


if not args.random_flag:
    set_random_seed(304, 2019)

data_info = load_data(args)
res_file = '../data/result/' + args.dataset + '_' + str(args.n_epoch) + '_' + str(args.lr) + '_' + str(
    args.l2_weight) + '_' + str(args.ng) + '_' + str(args.dim) + '_' + str(
    args.user_triple_set_size) + '_' + str(args.item_triple_set_size) + '_' + str(
    args.neighbor_size) + '_' + args.agg + '.txt'
res_writer = open(res_file, 'w', encoding='utf-8')
train(args, data_info, res_writer)
res_writer.close()
