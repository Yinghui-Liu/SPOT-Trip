import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import ones_


class GAT(nn.Module):
    '''
    The GAT class encapsulates the functionality of a Graph Attention Network.
    It is particularly useful in scenarios where the relationships between entities in a graph are complex
    and require attention mechanisms to model these relationships effectively
    '''
    def __init__(self, nfeat, nhid, dropout, alpha):
        """
        Initializes the dense version of Graph Attention Network (GAT) module.
        This class implements a GAT layer, a type of graph neural network layer
        that uses attention mechanisms to weight the influence of different nodes.
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.layer = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, poi_embs, entity_embs, adj):
        """
        Forward pass of the GAT layer.
        This method applies the GAT layer to POI and entity embeddings
        Returns:
            Tensor
        """
        x = F.dropout(poi_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward_relation(self, poi_embs, entity_embs, w_r, adj):
        """
        Forward pass of the GAT layer with relation embeddings.
        This extends this to include relation embeddings, allowing for more nuanced modeling of graph-based relationships
        Returns:
            Tensor
        """
        x = F.dropout(poi_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation(x, y, w_r, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphAttentionLayer(nn.Module):
    '''
    The `GraphAttentionLayer` class defines a layer in a graph attention network (GAT).
    This layer uses attention mechanisms to weight the influence of neighboring nodes differently.
    '''
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
        Initializes a Graph Attention Layer.
        This layer implements the attention mechanism in a graph attention network,
        allowing nodes to focus on different parts of their neighborhoods.
        """
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2*out_features, out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.leakyrelu = nn.ReLU()

    def forward_relation(self, poi_embs, entity_embs, relations, adj):
        """
        Forward pass with relation embeddings.
        The `forward_relation` method includes relation embeddings in the computation,
        allowing the model to consider relationships between entities.
        Adj:
            poi_embs: N, dim
            entity_embs: N, e_num, dim
            relations: N, e_num, r_dim
            adj: N, e_num
        Returns:
            Tensor
        """
        # N, e_num, dim
        Wh = poi_embs.unsqueeze(1).expand(entity_embs.size())
        # N, e_num, dim
        We = entity_embs
        a_input = torch.cat((Wh,We),dim=-1) # (N, e_num, 2*dim)
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(self.fc(a_input), relations).sum(-1) # N,e
        e = self.leakyrelu(e_input) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+poi_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward(self, poi_embs, entity_embs, adj):
        """
        Forward pass of the layer.
        The `forward` method computes the attention mechanism without relation embeddings.
        Returns:
            Tensor
        """
        Wh = torch.mm(poi_embs, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        We = torch.matmul(entity_embs, self.W) # entity_embs: (N, e_num, in_features), We.shape: (N, e_num, out_features)
        a_input = self._prepare_cat(Wh, We) # (N, e_num, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+poi_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        """
        Prepares concatenated vectors for attention score computation.
        Returns:
            Tensor
        """
        Wh = Wh.unsqueeze(1).expand(We.size()) # (N, e_num, out_features)
        return torch.cat((Wh, We), dim=-1) # (N, e_num, 2*out_features)


    def _prepare_attentional_mechanism_input(self, Wh):
        """
        Prepares the input for the attentional mechanism.
        Returns:
            Tensor
        """
        N = Wh.size()[0] # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
