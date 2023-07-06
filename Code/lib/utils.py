import numpy as np
import torch

def common_loss2(adj_1, adj_2):
    adj_1 = adj_1 - torch.eye(adj_1.shape[0]).cuda()
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    cost = torch.sum((adj_1 - adj_2) ** 2)
    cost = torch.exp(-cost)
    return cost


def common_loss(adj_1, adj_2):
    adj_1 = adj_1 * (1 - torch.eye(adj_1.shape[0]).cuda())
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    cost = torch.sum((adj_1 - adj_2) ** 2)
    cost = torch.exp(-cost)
    return cost

def dependence_loss(adj_1, adj_2):
    node_num = adj_1.shape[0]
    R = torch.eye(node_num) - (1/node_num) * torch.ones(node_num, node_num)
    adj_1 = adj_1 * (1 - torch.eye(adj_1.shape[0]).cuda())
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    K1 = torch.mm(adj_1, adj_1.T)
    K2 = torch.mm(adj_2, adj_2.T)
    RK1 = torch.mm(R.cuda(), K1)
    RK2 = torch.mm(R.cuda(), K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def loss_dependence2(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC
