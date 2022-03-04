import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CKGAT(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(CKGAT, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.W_GAT = nn.Parameter(torch.zeros(size=(self.dim, self.dim))) 
        self.a_GAT = nn.Parameter(torch.zeros(size=(3 * self.dim, 1)))
        self.leakyrelu = nn.LeakyReLU(self.ng)
        self.W_agg = nn.Parameter(torch.zeros(size=(self.dim, self.dim))) 
        self.w_agg = nn.Parameter(torch.zeros(size=(self.dim, 1)))
        self._init_weight()

    def forward(
            self,
            items: torch.LongTensor,
            user_triple_set: list,  
            item_triple_set: list,
            # h/r/t: [layers, batch_size, triple_set_size] nei_h/r/t: [layers, batch_size, triple_set_size, neighbor_size]
    ):
        user_embeddings = []  

        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])  
        # [batch_size, dim]
        user_embeddings.append(user_emb_0.mean(dim=1))  

        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i])  
            r_emb = self.relation_emb(user_triple_set[1][i])
            t_emb = self.entity_emb(user_triple_set[2][i])
            # [batch_size, dim]
            # user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)

            # [batch_size, triple_set_size, neighbor_size, dim]
            nei_h_emb = self.entity_emb(user_triple_set[3][i])  
            nei_r_emb = self.relation_emb(user_triple_set[4][i])
            nei_t_emb = self.entity_emb(user_triple_set[5][i])
            # [batch_size, dim]
            user_emb_i = self._knowledge_gat(nei_h_emb, nei_r_emb, nei_t_emb, t_emb) 
            user_embeddings.append(user_emb_i)

        item_embeddings = []  
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)  # e(origin)v
        item_embeddings.append(item_emb_origin)

        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])
            # [batch_size, dim]
            # item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)  # e(l)v
            # [batch_size, triple_set_size, neighbor_size, dim]
            nei_h_emb = self.entity_emb(item_triple_set[3][i])  
            nei_r_emb = self.relation_emb(item_triple_set[4][i])
            nei_t_emb = self.entity_emb(item_triple_set[5][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_gat(nei_h_emb, nei_r_emb, nei_t_emb, t_emb)  
            item_embeddings.append(item_emb_i)

        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
            # [batch_size, triple_set_size, dim]
            item_emb_0 = self.entity_emb(item_triple_set[0][0])
            # [batch_size, dim]
            item_embeddings.append(item_emb_0.mean(dim=1))  # e(0)v

        scores = self.predict(user_embeddings, item_embeddings)
        return scores

    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]  # e(0)u
        e_v = item_embeddings[0]  # e(origin)v

        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u), dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        elif self.agg == 'attention':
            e_u = self.agg_attention(user_embeddings)
            e_v = self.agg_attention(item_embeddings)
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        # [batch_size] = [batch_size, dim] * [batch_size, dim]
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)  # [batch_size]
        return scores

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        self.ng = args.ng  
        self.use_cuda = args.use_cuda

    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # init attention
        for layer in self.attention:
            if isinstance(layer, nn.Linear):  
                nn.init.xavier_uniform_(layer.weight)
        # init GAT
        nn.init.xavier_uniform_(self.W_GAT.data)
        nn.init.xavier_uniform_(self.a_GAT.data)
        nn.init.xavier_uniform_(self.W_agg.data)
        nn.init.xavier_uniform_(self.w_agg.data)

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb, r_emb), dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights, dim=-1)  # PI(ehi,ri)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)  # [bs,tss,1]*[bs,tss,dim]
        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)  # e(l)o
        return emb_i

    def _knowledge_gat(self, nei_h_emb, nei_r_emb, nei_t_emb, t_emb):
        # [batch_size, triple_set_size, neighbor_size, dim]=[batch_size, triple_set_size, neighbor_size, dim] * [dim, dim]
        Wh = torch.matmul(nei_h_emb, self.W_GAT)
        Wr = torch.matmul(nei_r_emb, self.W_GAT)
        Wt = torch.matmul(nei_t_emb, self.W_GAT)
        # [batch_size, triple_set_size, neighbor_size, 3*dim]
        hrt = torch.cat([Wh, Wr, Wt], dim=3)
        # [batch_size, triple_set_size, neighbor_size,1] = [batch_size, triple_set_size, neighbor_size, 3*dim] * [3*dim,1]
        # [batch_size, triple_set_size, neighbor_size] <-- [batch_size, triple_set_size, neighbor_size,1]
        pi = self.leakyrelu(torch.matmul(hrt, self.a_GAT).squeeze(3))
        # [batch_size, triple_set_size, neighbor_size]
        att = F.softmax(pi, dim=2)
        # [batch_size, triple_set_size, 1, dim] = [batch_size, triple_set_size, 1, neighbor_size] * [batch_size, triple_set_size, neighbor_size, dim]
        # [batch_size, triple_set_size, dim] <-- [batch_size, triple_set_size, 1, dim]
        nei_rep = torch.matmul(att.unsqueeze(2), nei_h_emb).squeeze(dim=2)
        # [batch_size, triple_set_size, dim] = [batch_size, triple_set_size, dim] * [dim, dim]
        emb_i = F.elu(torch.matmul(torch.add(nei_rep, t_emb), self.W_GAT))
        # [batch_size, dim] <-- [batch_size, triple_set_size, dim]
        emb_i = emb_i.sum(dim=1)  # e(l)o
        return emb_i

    def agg_attention(self, embeddings):
        if self.use_cuda:
            embeddings = list(map(lambda x: x.cuda(), embeddings))
        # [batch_size, L+1, dim] <-- [L+1, batch_size, dim]
        embeddings = torch.stack(embeddings, dim=1)
        # [batch_size, L+1, dim] = [batch_size, L+1, dim] * [dim, dim]
        Wemb = torch.matmul(embeddings, self.W_agg)
        # [batch_size, L+1, 1] = [batch_size, L+1, dim] * [dim, 1]
        # [batch_size, L+1] <-- [batch_size, L+1, 1]
        scores = torch.matmul(torch.tanh(Wemb), self.w_agg).squeeze(dim=2)
        # [batch_size, L+1]
        alpha = F.softmax(scores, dim=0)
        # [batch_size, 1, dim] = [batch_size, 1, L+1] * [batch_size, L+1, dim]
        e_agg = torch.matmul(alpha.unsqueeze(1), embeddings)
        # [batch_size, dim] <-- [batch_size, 1, dim]
        e_agg = e_agg.squeeze(1)
        return e_agg
