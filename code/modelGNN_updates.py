import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import scipy.sparse as sp

att_op_dict = {
    'sum': 'sum',
    'mul': 'mul',
    'concat': 'concat'
}

"""GatedGNN with residual connection"""
class ReGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu,
                 residual=True, att_op='mul', alpha_weight=1.0):
        super(ReGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.emb_encode = nn.Linear(feature_dim_size, hidden_size)
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size)
        self.z1 = nn.Linear(hidden_size, hidden_size)
        self.r0 = nn.Linear(hidden_size, hidden_size)
        self.r1 = nn.Linear(hidden_size, hidden_size)
        self.h0 = nn.Linear(hidden_size, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a)
        z1 = self.z1(x)
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a) + self.r1(x))
        # update embeddings
        h = self.act(self.h0(a) + self.h1(r * x))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs.float()
        adj = adj.float()
        mask = mask.float()
        x = self.dropout_encode(x)
        x = self.emb_encode(x)
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            if self.residual:
                x = x + self.gatedGNN(x, adj) * mask
            else:
                x = self.gatedGNN(x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask

        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings

"""GCNs with residual connection"""
class ReGCN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu,
                 residual=True, att_op="mul", alpha_weight=1.0):
        super(ReGCN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.gnnlayers.append(GraphConvolution(feature_dim_size, hidden_size, dropout, act=act))
            else:
                self.gnnlayers.append(GraphConvolution(hidden_size, hidden_size, dropout, act=act))
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act

    def forward(self, inputs, adj, mask):
        x = inputs.float()
        adj = adj.float()
        mask = mask.float()
        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](x, adj) * mask
            else:
                if self.residual:
                    x = x + self.gnnlayers[idx_layer](x, adj) * mask
                else:
                    x = self.gnnlayers[idx_layer](x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings


"""GatedGNN"""
class GGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu):
        super(GGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size)
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size)
        self.z1 = nn.Linear(hidden_size, hidden_size)
        self.r0 = nn.Linear(hidden_size, hidden_size)
        self.r1 = nn.Linear(hidden_size, hidden_size)
        self.h0 = nn.Linear(hidden_size, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a)
        z1 = self.z1(x)
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a) + self.r1(x))
        # update embeddings
        h = self.act(self.h0(a) + self.h1(r * x))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs.float()
        adj = adj.float()
        mask = mask.float()
        x = self.dropout_encode(x)
        x = self.emb_encode(x)
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            x = self.gatedGNN(x, adj) * mask
        return x


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x.float(), self.weight.float())
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


weighted_graph = False
print('using default unweighted graph')


# build graph function
def _uni_graph(doc_words, window_size=3):
    """Build the co-occurrence adjacency for one token sequence.

    Shared by build_graph() and build_graph_index() so that the static and the
    contextual feature paths are guaranteed to produce *identical* graph
    structure -- only the node features differ between them.

    Returns (adj, doc_word_id_map) where the map sends a token id to its node.
    """
    doc_len = len(doc_words)
    doc_vocab = list(set(doc_words))
    doc_nodes = len(doc_vocab)

    doc_word_id_map = {}
    for j in range(doc_nodes):
        doc_word_id_map[doc_vocab[j]] = j

    # sliding windows
    windows = []
    if doc_len <= window_size:
        windows.append(doc_words)
    else:
        for j in range(doc_len - window_size + 1):
            windows.append(doc_words[j: j + window_size])

    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p_id = window[p]
                word_q_id = window[q]
                if word_p_id == word_q_id:
                    continue
                word_pair_key = (word_p_id, word_q_id)
                # word co-occurrences as weights
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # bi-direction
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.

    row = []
    col = []
    weight = []
    for key in word_pair_count:
        row.append(doc_word_id_map[key[0]])
        col.append(doc_word_id_map[key[1]])
        weight.append(word_pair_count[key] if weighted_graph else 1.)

    adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    return adj, doc_word_id_map


def build_graph(shuffle_doc_words_list, word_embeddings, window_size=3):
    """Graph with *static* node features taken from the embedding table."""
    x_adj = []
    x_feature = []

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        adj, doc_word_id_map = _uni_graph(doc_words, window_size)
        x_adj.append(adj)

        features = []
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[k])
        x_feature.append(features)

    return x_adj, x_feature


def build_graph_index(shuffle_doc_words_list, window_size=3):
    """Same adjacency as build_graph(), but returns node indices per position.

    Node features are *not* materialised here: the caller gathers them from
    the encoder's contextual hidden states in torch, which is what keeps the
    encoder (and therefore the LoRA adapters) in the autograd graph.

    Returns (x_adj, node_ids) where node_ids[i][p] is the graph node that
    token position p of sample i belongs to.
    """
    x_adj = []
    node_ids = []

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        adj, doc_word_id_map = _uni_graph(doc_words, window_size)
        x_adj.append(adj)
        node_ids.append(
            np.array([doc_word_id_map[w] for w in doc_words], dtype=np.int64)
        )

    return x_adj, node_ids


def _text_graph(doc_len, window_size=3):
    """Positional adjacency for one sequence: node i is token position i."""
    row = []
    col = []
    weight = []

    if doc_len > window_size:
        for j in range(doc_len - window_size + 1):
            for p in range(j + 1, j + window_size):
                for q in range(j, p):
                    row.append(p)
                    col.append(q)
                    weight.append(1.)
                    #
                    row.append(q)
                    col.append(p)
                    weight.append(1.)
    else:  # doc_len < window_size
        for p in range(1, doc_len):
            for q in range(0, p):
                row.append(p)
                col.append(q)
                weight.append(1.)
                #
                row.append(q)
                col.append(p)
                weight.append(1.)

    adj = sp.csr_matrix((weight, (row, col)), shape=(doc_len, doc_len))
    if weighted_graph == False:
        adj[adj > 1] = 1.
    return adj


def build_graph_text_index(shuffle_doc_words_list, window_size=3):
    """Positional variant: node i is token position i, so the mapping is identity."""
    x_adj = [_text_graph(len(doc), window_size)
             for doc in shuffle_doc_words_list]
    node_ids = [np.arange(len(doc), dtype=np.int64)
                for doc in shuffle_doc_words_list]
    return x_adj, node_ids


# another way to build graph from text
def build_graph_text(shuffle_doc_words_list, word_embeddings, window_size=3):

    # print('using window size = ', window_size)
    x_adj = []
    x_feature = []
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        doc_len = len(doc_words)

        row = []
        col = []
        weight = []
        features = []

        if doc_len > window_size:
            for j in range(doc_len - window_size + 1):
                for p in range(j + 1, j + window_size):
                    for q in range(j, p):
                        row.append(p)
                        col.append(q)
                        weight.append(1.)
                        #
                        row.append(q)
                        col.append(p)
                        weight.append(1.)
        else:  # doc_len < window_size
            for p in range(1, doc_len):
                for q in range(0, p):
                    row.append(p)
                    col.append(q)
                    weight.append(1.)
                    #
                    row.append(q)
                    col.append(p)
                    weight.append(1.)

        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_len, doc_len))
        if weighted_graph == False:
            adj[adj > 1] = 1.
        x_adj.append(adj)
        #
        for word in doc_words:
            feature = word_embeddings[word]
            features.append(feature)
        x_feature.append(features)

    return x_adj, x_feature
