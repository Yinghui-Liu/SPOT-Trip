import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random
from utils import _L2_loss_mean
from GAT import GAT
from einops import repeat
from torchdiffeq import odeint
import torchquad
from torch.distributions import Normal, Independent
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from trainer import top_np_recommendation

class Encoder(nn.Module):
    """Encoder mapping context sequences to parameters of the posterior q(z1)."""

    def __init__(
        self, poi_size,
        d_z: int,
        d_model: int, n_attn_heads: int, n_tf_layers: int, dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.time_proj = nn.Linear(1, d_model, bias=False)
        self.space_proj = nn.Linear(2, d_model, bias=False)
        self.poi_proj = nn.Linear(d_model, d_model, bias=False)
        self.poi_emb = POIEmbeddings(poi_size, d_model)
        self.transformer_stack = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_attn_heads,
                dim_feedforward=2 * d_model,
                batch_first=True,
                dropout=dropout_prob,
            ) for _ in range(n_tf_layers)
        ])

        self.gamma_proj = nn.Linear(d_model, d_z)
        self.tau_proj = nn.Linear(d_model, d_z)

        self.agg_token = nn.Parameter(torch.empty((1, 1, d_model)))
        nn.init.xavier_uniform_(self.agg_token)

    def forward(self, d_t, d_l, d_emb, d_pad):
        """Maps context sequences `x` to parameters of the posterior q(z1)."""

        t_emb = self.time_proj(d_t.to(torch.float32).unsqueeze(-1))
        coords_emb = self.space_proj(d_l.to(torch.float32))
        # poi_emb = self.poi_proj(d_emb)
        poi_emb = self.poi_emb(d_emb)

        x = torch.cat(
            [
                t_emb + coords_emb + poi_emb,
                repeat(self.agg_token, "() () d -> b () d", b=d_t.shape[0]),
            ],
            dim=1,
        )
        # x = torch.cat(
        #     [
        #         t_emb + poi_emb,
        #         repeat(self.agg_token, "() () d -> b () d", b=d_t.shape[0]),
        #     ],
        #     dim=1,
        # )

        for layer in self.transformer_stack:
            x = layer(x, src_key_padding_mask=d_pad)

        x = x[:, -1, :]

        return x, self.gamma_proj(x), torch.nn.functional.softplus(self.tau_proj(x))  # 更稳定

def _nearest_interpolate(t_eval, t, z, ind_left, ind_right):
    dist_left = torch.abs(t_eval - t[ind_left])
    dist_right = torch.abs(t_eval - t[ind_right])
    nearer_right = dist_right < dist_left
    return torch.where(nearer_right.unsqueeze(1), z[ind_right], z[ind_left])


def _linear_interpolate(t_eval, t, z, ind_left, ind_right):
    t_left = t[ind_left]
    t_right = t[ind_right]
    weight_right = (t_eval - t_left) / (t_right - t_left + 1e-3)
    weight_left = 1 - weight_right
    return weight_left.unsqueeze(1) * z[ind_left] + weight_right.unsqueeze(1) * z[ind_right]

def interpolate(t_eval, t, z, method: str = "nearest"):
    """
    Interpolates values at specified evaluation points.

    Args:
        t_eval (Tensor): The evaluation time points, shape (n,).
        t (Tensor): The trajectory time points, shape (time,).
        z (Tensor): The trajectory values at time points `t`, shape (time, d_z).
        method (str, optional): The interpolation method ('nearest' or 'linear'). Defaults to 'nearest'.

    Returns:
        Tensor: Interpolated values at `t_eval`.
    """
    if method not in {"nearest", "linear"}:
        raise ValueError(f"Interpolation method {method} is not supported.")

    ind_right = torch.searchsorted(t, t_eval) # 查找 t_eval 在时间序列 t 中的插入位置，返回的 ind_right 是右侧的索引
    ind_left = ind_right - 1
    ind_left.clamp_(min=0)
    ind_right.clamp_(max=len(t) - 1)

    if method == "nearest":
        return _nearest_interpolate(t_eval, t, z, ind_left, ind_right)
    else:  # method == "linear"
        return _linear_interpolate(t_eval, t, z, ind_left, ind_right)

def kl_norm_norm(mu0, mu1, sig0, sig1):
    """Calculates KL divergence between two K-dimensional Normal
        distributions with diagonal covariance matrices.

    Args:
        mu0: Mean of the first distribution. Has shape (*, K).
        mu1: Mean of the second distribution. Has shape (*, K).
        sig0: Diagonal of the covatiance matrix of the first distribution. Has shape (*, K).
        sig1: Diagonal of the covatiance matrix of the second distribution. Has shape (*, K).

    Returns:
        KL divergence between the distributions. Has shape (*, 1).
    """
    assert mu0.shape == mu1.shape == sig0.shape == sig1.shape, (f"{mu0.shape=} {mu1.shape=} {sig0.shape=} {sig1.shape=}")
    a = (sig0 / sig1).pow(2).sum(-1, keepdim=True)
    b = ((mu1 - mu0).pow(2) / sig1**2).sum(-1, keepdim=True)
    c = 2 * (torch.log(sig1) - torch.log(sig0)).sum(-1, keepdim=True)
    kl = 0.5 * (a + b + c - mu0.shape[-1])
    return kl

def create_mlp(
        input_size,
        output_size,
        hidden_size,
        num_hidden_layers,
        activation_func,
        use_layer_norm=False,
        use_dropout=False,
        dropout_prob=0.5,
):
    """
    Create MLP with optional layer normalization and dropout.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        hidden_size (int): The size of the hidden layers.
        num_hidden_layers (int): The number of hidden layers.
        activation_func (function): The nonlinear activation function to use.
        use_layer_norm (bool): Whether to use layer normalization (default: False).
        use_dropout (bool): Whether to use dropout (default: False).
        dropout_prob (float): Dropout probability, used if use_dropout is True (default: 0.5).

    Returns:
        nn.Sequential: The constructed MLP model.
    """
    layers = []
    for i in range(num_hidden_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, hidden_size))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation_func())
        if use_dropout:
            layers.append(nn.Dropout(dropout_prob))

    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)

class DynamicsFunction(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, t, z):
        return self.f(z)


class DynamicTimeGenerator(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(DynamicTimeGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
        self.fc_interval = nn.Linear(hidden_dim, 1)
        self.init_input = nn.Parameter(torch.zeros(1, input_dim), requires_grad=False)

    def forward(self, context, num_pred_list):
        batch_size = context.size(0)
        device = context.device

        full_times_list = []
        seq_lengths = []

        for i in range(batch_size):
            n_pred = num_pred_list[i]
            hidden = context[i:i + 1]
            rnn_input = self.init_input
            cum_time = torch.zeros(1, device=device)
            times = []
            for _ in range(n_pred):
                hidden = self.rnn_cell(rnn_input, hidden)
                delta = F.softplus(self.fc_interval(hidden))
                cum_time = cum_time + delta.squeeze(1)
                times.append(cum_time.clone())
            if len(times) > 0:
                times_tensor = torch.cat(times, dim=0)
            else:
                times_tensor = torch.tensor([], device=device)
            if times_tensor.numel() > 0:
                times_tensor = times_tensor / (times_tensor[-1] + 1e-6)
            full_time = torch.cat(
                [torch.tensor([0.0], device=device), times_tensor, torch.tensor([1.0], device=device)], dim=0)
            full_times_list.append(full_time)
            seq_lengths.append(full_time.numel())

        max_len = max(seq_lengths)
        full_times_padded = torch.zeros(batch_size, max_len, device=device)
        for i, t_seq in enumerate(full_times_list):
            length = t_seq.numel()
            full_times_padded[i, :length] = t_seq

        return full_times_padded

class ContinuousDecoder(nn.Module):
    """Maps latent state z(t) and spatial coordinate x to u(t, x).

    Attributes:
        d_z (int): Dimensionality of the latent state.
        d_x (int): Dimensionality of the spatial coodinates.
        d_u (int): Dimensionality of the latent spatiotemporal state.
        f (Module): Mapping from (z(t), x) to u(t, x).
    """
    def __init__(self, d_z, d_x, f, interp_method):
        super().__init__()
        # self.space_proj = nn.Linear(d_x, d_z, bias=False)
        self.f = f # mlp
        self.interp_method = interp_method

    def forward(self, t_eval, t, z):
        """Evaluates the latent spatiotemporal state u(t, x) for a single trajectory t, z.

        Args:
            t_eval: Evaluation time points, has shape (n, ).
            t: Trajectory time points, has shape (time, ).
            z: Trajectory values at time points `t`, has shape (time, d_z).

        Returns:
            Latent spatiotemporals state at (t_eval, x_eval). Has shape (n, d_u).
        """
        if t_eval.ndim != 1 or t.ndim != 1:
            raise ValueError("t and t_eval should be a 1-dimensional arrays.")
        if z.ndim != 2:
            raise ValueError("z should be a 2-dimensional arrays.")
        if t.shape[0] != z.shape[0]:
            raise ValueError("t and z must have matching first dimension.")

        z_eval = interpolate(t_eval, t, z, method=self.interp_method)
        # return self.f(z_eval + self.space_proj(x_eval))
        return self.f(z_eval)
class IntensityCorrection(nn.Module):
    def __init__(self, val=0):
        super().__init__()
        self.val = val

    def forward(self, x):
        # return torch.pow(x, 2) + self.val
        return torch.exp(x) + self.val

class POIEmbeddings(nn.Module):
    def __init__(self, poi_size, poi_embed_dim):
        super(POIEmbeddings, self).__init__()
        self.emb = nn.Embedding(poi_size, poi_embed_dim)

    def forward(self, traj):
        x = self.emb(traj)
        return x

# =================Transformer framework================== #
class TransformerModel(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.3):
        super(TransformerModel, self).__init__()

        # self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src = src * math.sqrt(self.embed_size)
        # src = self.pos_encoder(src)
        x = self.transformer_encoder(src)

        return x


class Recommender(nn.Module):
    def __init__(self, out_dim, poi_size):
        super(Recommender, self).__init__()
        self.fc = nn.Linear(out_dim, poi_size)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, outputs):
        x = self.fc(outputs)
        x = self.leaky_relu(x)
        return x


# ============================Penalty=============================== #
class Drifting(nn.Module):
    def __init__(self, beta):
        super(Drifting, self).__init__()
        self.beta = beta

    def forward(self, fix_outputs, region_mask):

        batch_size, seq_len, _ = fix_outputs.size()
        max_num_moves = seq_len - 1
        total_similarity = 0.0

        count = 0

        for num_moves in range(1, max_num_moves + 1):
            for i in range(batch_size):
                valid_indices = region_mask[i]  # [poi_size]
                for t in range(seq_len - num_moves):
                    vec1 = fix_outputs[i, t, :][valid_indices]
                    vec2 = fix_outputs[i, t + num_moves, :][valid_indices]
                    sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
                    total_similarity += sim.item()
                    count += 1
        avg_similarity = torch.tensor(total_similarity / count)
        # for num_moves in range(1, max_num_moves + 1):
        #     shift_outputs = fix_outputs[:, num_moves:]
        #     similarity = F.cosine_similarity(fix_outputs[:, :-num_moves], shift_outputs, dim=-1)
        #     total_similarity += similarity.mean()
        # avg_similarity = total_similarity / (seq_len - 1)
        repetition_penalty_loss = -torch.log(1 - 0.5 * (avg_similarity + 1)) * self.beta

        return repetition_penalty_loss


class Guiding(nn.Module):
    def __init__(self, out_dim, poi_size):
        super(Guiding, self).__init__()
        self.predictor = Recommender(out_dim, poi_size)
        # self.confidence = nn.Linear(poi_size, 1)

    def forward(self, outputs, AM, PM):
        fix_outputs = self.predictor(outputs)  # [b,l,d] -> [b,l,v]
        clipped_PM = PM[:, :fix_outputs.shape[1]]  # [v,l_max] -> [v,l]
        clipped_outputs = fix_outputs * (clipped_PM.T.unsqueeze(0).expand(fix_outputs.shape[0], -1, -1))  # [b,l,v]

        return clipped_outputs

# Construct total framework(AR-Trip)
class SPOTModel(nn.Module):
    def __init__(self, args, poi_size, region_poi,
                 max_length_venue_id=100, d_model=128, n_head=4, num_encoder_layers=1, n_tf_layers=4, d_z=128, kg_dataset=None):

        super(SPOTModel, self).__init__()
        # initial hyperparameter
        self.hidden_size = d_model
        self.args = args
        # model setting
        self.poi_embedding = POIEmbeddings(poi_size, self.hidden_size)
        self.poi_size = poi_size
        self.pos_emb = nn.Embedding(max_length_venue_id, self.hidden_size)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
            nn.SiLU()
        )
        if self.args.kg:
            self.kg_dataset = kg_dataset
            self.n_entities = self.kg_dataset.entity_count
            self.n_relations = self.kg_dataset.relation_count
            self.entity_embedding = nn.Embedding(self.n_entities + 1, d_model)
            self.relations_embedding = nn.Embedding(self.n_relations + 1, d_model)
            self.kg_dict, self.poi2relations = self.kg_dataset.get_kg_dict(self.poi_size)
            self.gat = GAT(self.hidden_size, self.hidden_size, dropout=0.4, alpha=0.2).train()
            if self.args.trans == 'transr':
                self.projection_matrix = nn.Linear(self.hidden_size, self.args.projection_dim)

        self.encoder = Encoder(poi_size, d_z, d_model, n_head, n_tf_layers)
        self.dyf = DynamicsFunction(
            f=create_mlp(input_size=args.hidden_size,
            output_size=args.hidden_size,
            hidden_size=args.dyn_latent_dim,
            num_hidden_layers=args.dyn_hid_layers,
            activation_func=nn.GELU))
        self.time_generator = DynamicTimeGenerator(self.hidden_size, self.hidden_size)
        self.lm = nn.Sequential(
            create_mlp(
                input_size=args.hidden_size,
                output_size=1,
                hidden_size=args.lm_latent_dim,
                num_hidden_layers=args.lm_hid_layers,
                activation_func=nn.GELU,
            ),
            IntensityCorrection(0.0000001),
        )

        self.transformer_encoder = TransformerModel(embed_size=self.hidden_size * 2, nhead=n_head,
                                                    nhid=2048, nlayers=num_encoder_layers)
        self.infer_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU()
        )
        if self.args.ode and self.args.s_infer:
            self.predictor = Recommender(self.hidden_size * 4, poi_size)
        elif self.args.ode or self.args.s_infer:
            self.predictor = Recommender(self.hidden_size * 3, poi_size)
        else:
            self.predictor = Recommender(self.hidden_size * 2, poi_size)
        self.region_poi = region_poi
        self.region_embedding = nn.Embedding(len(self.region_poi), self.hidden_size)
        self.region_masks = {}
        for region, poi_list in region_poi.items():
            mask = torch.zeros(poi_size, dtype=torch.bool)
            mask[list(poi_list)] = True
            self.region_masks[region] = mask
        self.head_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.tail_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index during loss calculation

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        Calculates the loss for the model using the TransE approach.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        # Each sample corresponds to an index of a relation type, and embedding_relation converts the index of each relation type into the corresponding embedding vector.
        r_embed = self.relations_embedding(r)
        h_embed = self.poi_embedding(h) # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embedding(pos_t) # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embedding(neg_t) # (kg_batch_size, entity_dim)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1) # (kg_batch_size) As per the formula f_d in the paper.
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1) # (kg_batch_size)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        # This value can be considered as the "energy" of the input samples.
        # This code is typically used for calculating regularization terms in the loss function.
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def calc_kg_loss_transR(self, h, r, pos_t, neg_t):
        """
        Calculates the loss for the model using the TransR approach.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        r_embed = self.projection_matrix(self.relations_embedding(r))
        h_embed = self.projection_matrix(self.poi_embedding(h))
        pos_t_embed = self.projection_matrix(self.entity_embedding(pos_t))
        neg_t_embed = self.projection_matrix(self.entity_embedding(neg_t))
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def calc_kg_loss_SEEK(self, h, r, pos_t, neg_t):
        """
        Calculates the loss using the SEEK approach for knowledge graph embeddings.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        # Each sample corresponds to an index of a relation type, and the embedding_relation converts the index of each relation type into the corresponding embedding vector.
        r_embed = self.relations_embedding(r)        # (kg_batch_size, relation_dim)
        h_embed = self.poi_embedding(h)               # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embedding(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embedding(neg_t)      # (kg_batch_size, entity_dim)

        k_num = self.args.segments
        rank = int(self.hidden_size / k_num)
        h = [h_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
        h = tuple(h)
        r = [r_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
        r = tuple(r)
        pos_t = [pos_t_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
        pos_t = tuple(pos_t)
        neg_t = [neg_t_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
        neg_t = tuple(neg_t)
        pos_tmp = 0
        neg_tmp = 0

        for x in range(k_num):
            for y in range(k_num):
                s = -1 if x % 2 != 0 and x + y >= k_num else 1
                w = y if x % 2 == 0 else (x + y) % k_num
                pos_tmp += s * r[x] * h[y] * pos_t[w]
                neg_tmp += s * r[x] * h[y] * neg_t[w]
        pos_score = torch.sum(pos_tmp, 1)
        neg_score = torch.sum(neg_tmp, 1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        # This value can be considered as the "energy" of the input samples.
        # This code is typically used for calculating regularization terms in the loss function.
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    def _alias(self, bids, n_poi):
        """
        Creates an alias tensor mapping original POI indices to a new set of indices.
        Returns:
            torch.Tensor
        """
        alias = torch.zeros(n_poi).long()
        for idx, b in enumerate(bids):
            alias[b] = idx
        alias = alias.to(self.args.device)
        return alias

    def _avg_pooling(self, ck, emb):
        """
        Applies average pooling to embeddings.
        Returns:
            Tensor
        """
        emb_sum = torch.sum(emb, axis=1)
        row_count = torch.sum(ck != 0, axis=-1)
        emb_agg = emb_sum / row_count.unsqueeze(1).expand_as(emb_sum)
        return emb_agg

    def _relation(self, emb, r, rel_embs, mode='transr'):
        """
        Applies a relation transformation to the embeddings, Dynamic Mapping.
        Args:
            o_emb: b x h
            d_emb: b x l x h
        Returns:
            Tensor
        """
        if mode == 'transr':
            relation = rel_embs[r].view(-1, self.hidden_size, self.hidden_size) # b x (h x h)
            if len(emb.shape) == 2:
                emb_r = torch.bmm(emb.unsqueeze(1), relation).squeeze(1)
                return emb_r
            elif len(emb.shape) == 3:
                emb_r = torch.matmul(emb.unsqueeze(2), relation.unsqueeze(1).expand(-1, emb.size(1), -1, -1)).squeeze(2)
                return emb_r
        if mode == 'transd':
            relation = rel_embs[r] # b x h Embeddings of 64 regions (cities) visited by users
            if len(emb.shape) == 2:
                # b x h x h matrix multiplication
                # equivalent to the embedding weights of the region (city) multiplied by the node embedding that has passed through a linear layer.
                trans_mat = torch.matmul(relation.unsqueeze(2), self.head_linear(emb).unsqueeze(1)) # b x h x h
                # torch.bmm() might be faster, but both are matrix-level multiplication
                emb_r = torch.bmm(emb.unsqueeze(1), trans_mat).squeeze(1)
                return emb_r
            elif len(emb.shape) == 3:
                # b x h x h (64, 13, 128, 1) * (64, 13, 1, 128)
                trans_mat = torch.matmul(relation.view(relation.size(0), 1, -1, 1).expand(-1, emb.size(1), -1, -1), self.tail_linear(emb).unsqueeze(2)) # b x h x h
                emb_r = torch.matmul(emb.unsqueeze(2), trans_mat).squeeze(2)
                return emb_r
        if mode == 'transe':
            return emb

    def drop_edge_random(self, poi2entities, p_drop, padding):
        """
        Randomly drops edges from the POI to entity mappings.
        Returns:
            dict
        """
        res = dict()
        for item, es in poi2entities.items():
            new_es = list()
            for e in es.tolist():
                if (random.random() > p_drop):
                    new_es.append(e)
                else:
                    new_es.append(padding)
            res[item] = torch.IntTensor(new_es).to(self.args.device)
        return res

    def get_kg_views(self):
        """
        Generates two views of the knowledge graph by randomly dropping edges.
        Returns:
            tuple
        """
        kg = self.kg_dict
        view1 = self.drop_edge_random(kg, self.args.kg_p_drop, self.n_entities)
        view2 = self.drop_edge_random(kg, self.args.kg_p_drop, self.n_entities)
        return view1, view2

    def cal_poi_embedding_mean(self, kg: dict):
        """
        Calculates the mean embeddings of POIs based on their associated entities.
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
        entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
        # padding is zero
        entity_embs = entity_embs * padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # poi_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # poi_num, emb_dim
        return poi_embs+entity_embs_mean

    def cal_poi_embedding_gat(self, kg:dict):
        """
        Calculates the POI embeddings using a Graph Attention Network (GAT) based on the associated entities.
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
        entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
        # poi_num, entity_num_each
        padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
        return self.gat(poi_embs, entity_embs, padding_mask)

    def cal_poi_embedding_rgat(self, kg:dict):
        """
        Calculates POI embeddings using a Relational Graph Attention Network (RGAT).
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
        poi_relations = torch.stack(list(self.poi2relations.values()))
        entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
        relation_embs = self.relations_embedding(poi_relations) # poi_num, entity_num_each, emb_dim
        padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
        return self.gat.forward_relation(poi_embs, entity_embs, relation_embs, padding_mask)

    def cal_poi_embedding_from_kg(self, kg: dict):
        """
        Calculates POI embeddings based on the specified knowledge graph convolution method.
        Returns:
            Tensor
        """
        if kg is None:
            kg = self.kg_dict

        if(self.args.kgcn=="GAT"):
            return self.cal_poi_embedding_gat(kg)
        elif self.args.kgcn=="RGAT":
            return self.cal_poi_embedding_rgat(kg)
        elif(self.args.kgcn=="MEAN"):
            return self.cal_poi_embedding_mean(kg)
        elif(self.args.kgcn=="NO"):
            return self.poi_embedding.weight

    def get_ui_views_weighted(self, poi_stabilities, stab_weight):
        """
        Calculates weighted POI views based on stability scores.
        Returns:
            Tensor
        """
        # kg probability of keep
        poi_stabilities = torch.exp(poi_stabilities)
        kg_weights = (poi_stabilities - poi_stabilities.min()) / (poi_stabilities.max() - poi_stabilities.min())
        # Replace elements in kg_weights less than or equal to 0.3 with 0.3, keep elements greater than 0.3 unchanged.
        kg_weights = kg_weights.where(kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)
        weights = (1-self.args.ui_p_drop)/torch.mean(stab_weight*kg_weights)*(stab_weight*kg_weights)
        # weights = weights.where(weights>0.3, torch.ones_like(weights) * 0.3)
        # Replace elements in weights greater than or equal to 0.95 with 0.95, keep elements less than 0.95 unchanged.
        weights = weights.where(weights<0.95, torch.ones_like(weights) * 0.95)
        # Perform Bernoulli sampling to get a mask tensor poi_mask of the same dimension as weights,
        # where the probability of an element being True is the corresponding value in weights.
        # Values are chosen as 1 or 0 with probabilities p and 1-p, respectively.
        poi_mask = torch.bernoulli(weights).to(torch.bool)
        # drop
        poi_mask.requires_grad = False
        return poi_mask

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Calculates the similarity between two tensors.
        Returns:
            Tensor
        """
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1,z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def poi_kg_stability(self, view1, view2):
        """
        Computes the stability of POI embeddings across two views of the knowledge graph.
        Returns:
            Tuple
        """
        kgv1_ro = self.cal_poi_embedding_from_kg(view1)
        kgv2_ro = self.cal_poi_embedding_from_kg(view2)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return kgv1_ro, kgv2_ro, sim


    def get_views(self, aug_side="both"):
        """
        Generates augmented views for contrastive learning.
        Returns:
            Dict
        """
        # drop (epoch based)
        # kg drop -> 2 views -> view similarity for item
        # Randomly remove tail entities and fill in the removed parts.
        kgv1, kgv2 = self.get_kg_views()
        # [item_num]
        kgv1, kgv2, stability = self.poi_kg_stability(kgv1, kgv2)  # Calculate consistency
        kgv1 = kgv1.to(self.args.device)
        kgv2 = kgv2.to(self.args.device)
        stability = stability.to(self.args.device)
        # item drop -> 2 views
        # Delete the user-item interaction edges (deleting edges with item nodes as index) from the interaction graph.
        v1_mask = self.get_ui_views_weighted(stability, 1)
        # uiv2 = self.ui_drop_random(world.ui_p_drop)
        v2_mask = self.get_ui_views_weighted(stability, 1)

        contrast_views = {
            "kgv1": kgv1,
            "kgv2": kgv2,
            "uiv1": v1_mask,
            "uiv2": v2_mask
        }
        return contrast_views

    def info_nce_loss_overall(self, z1, z2):
        """
        Calculates the InfoNCE loss, a contrastive loss used for learning efficient embeddings.
        Returns:
            torch.Tensor
        """
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)  # (batch_size * 2, 1)
        logits = logits / self.tau

        loss = criterion(logits, labels)
        return loss

    def pad_with_embedding(self, seq_list, pad_vector):
        # seq_list: list of tensors [seq_len, d]
        # pad_vector: tensor of shape [d]
        max_len = max(seq.shape[0] for seq in seq_list)
        padded_list = []
        for seq in seq_list:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                pad_tensor = pad_vector.unsqueeze(0).expand(pad_len, -1)
                padded_seq = torch.cat([seq, pad_tensor], dim=0)
            else:
                padded_seq = seq
            padded_list.append(padded_seq)
        return torch.stack(padded_list, dim=0)

    def forward(self, o_ck, query, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg, target_seq=None):
        batch_size, seq_length = query.size()
        # region_repr = self.region_embedding.weight
        region_mask = torch.stack([self.region_masks[int(r)] for r in d_rg], dim=0)
        region_mask_uns = region_mask.unsqueeze(1).expand(-1, query.size(1), -1).to(query.device)
        # initialize
        if self.args.kg:
            poi_embedding = self.cal_poi_embedding_from_kg(self.kg_dict)
            query_emb = poi_embedding[query]  # [b,l,d]
            o_emb = poi_embedding[o_ck]
            d_target_emb = poi_embedding[d_ck]
            pad_vec = poi_embedding[0]
        else:
            query_emb = self.poi_embedding(query)  # [b,l,d]
            o_emb = self.poi_embedding(o_ck)
            d_target_emb = self.poi_embedding(d_ck)
            pad_vec = self.poi_embedding.emb.weight[0]

        if self.args.ode:
            u_o_emb_d, gamma, tau = self.encoder(o_t, o_l, o_ck, o_pad)
            z_0 = gamma + tau * torch.randn_like(tau)
            dynamic_d_emb = self.encoder.time_proj(d_t.to(torch.float32).unsqueeze(-1)) + self.encoder.space_proj(d_l) + self.encoder.poi_emb(d_ck)
            # dynamic_d_emb = self.encoder.time_proj(d_t.to(torch.float32).unsqueeze(-1)) + self.encoder.poi_emb(d_ck)
            P_D = []
            process_loglik = torch.tensor([0.0], device=self.args.device, dtype=torch.float32)
            obs_loglik = torch.tensor([0.0], device=self.args.device, dtype=torch.float32)
            for j in range(batch_size):
                valid_idx = torch.nonzero(d_pad[j], as_tuple=True)[0][:-1]
                n_pred = len(valid_idx) - 2
                if target_seq is not None:
                    gt_times_unif = d_t[j][valid_idx].to(torch.float32)
                    # print("gt_times_unif:", gt_times_unif)
                    z_unif_j = odeint(self.dyf, z_0[j].unsqueeze(0), gt_times_unif,
                                      rtol=self.args.rtol, atol=self.args.atol, method=self.args.solver)
                else:
                    s_unif = torch.linspace(0, 1, n_pred + 2, device=self.args.device, dtype=torch.float32)
                    z_unif_j = odeint(self.dyf, z_0[j].unsqueeze(0), s_unif,
                                      rtol=self.args.rtol, atol=self.args.atol, method=self.args.solver)
                u_hat = z_unif_j.transpose(0, 1).squeeze(0)
                P_D.append(u_hat)
                if target_seq is not None:
                    lm_hat = self.lm(u_hat)
                    process_loglik += torch.sum(torch.log(lm_hat))
                    f_values = self.lm(z_unif_j).squeeze(-1).squeeze(-1)

                    integrated_value = torch.trapz(f_values, gt_times_unif)

                    process_loglik -= integrated_value

                    v = dynamic_d_emb[j][valid_idx]
                    obs_loglik += Normal(u_hat, self.args.sig_v).log_prob(v).sum()
            if target_seq is not None:
                kl_qp = 2 * kl_norm_norm(gamma, torch.zeros_like(gamma), tau, torch.ones_like(tau)).sum()
                elbo_loss = - (obs_loglik + process_loglik - kl_qp)
            P_D = self.pad_with_embedding(P_D, pad_vec)
        if self.args.s_infer:
            u_o_emb_s = self._avg_pooling(o_ck, o_emb)
            infer = self.infer_layer(u_o_emb_s)
            if target_seq is not None:
                u_d_emb_s = self._avg_pooling(d_ck, d_target_emb)
                infer_loss = torch.norm(infer - u_d_emb_s, p=2, dim=-1).mean()
            P_S = infer.unsqueeze(1).expand_as(d_target_emb)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embedded = self.pos_emb(position_ids)
        model_input = torch.cat([query_emb, position_embedded], dim=2)
        encoder_output = self.transformer_encoder(model_input)

        if self.args.ode and self.args.s_infer:
            encoder_output = torch.cat([encoder_output, P_D, P_S], dim=2)
        elif self.args.ode:
            encoder_output = torch.cat([encoder_output, P_D], dim=2)
        elif self.args.s_infer:
            encoder_output = torch.cat([encoder_output, P_S], dim=2)
        poi_output = self.predictor(encoder_output)
        masked_poi_output = poi_output.masked_fill(~region_mask_uns, -1e9)
        if target_seq is not None:
            loss = self.criterion(masked_poi_output.view(-1, self.poi_size), d_ck.flatten())
            if self.args.ode:
                loss += elbo_loss.sum()
            if self.args.s_infer:
                loss += infer_loss
            return loss
        else:
            # _, predicted_ids = torch.max(masked_poi_output, dim=-1)
            guidance_similarity_ratio, guidance_candidate_ids = torch.topk(masked_poi_output,
                                                                           k=masked_poi_output.shape[1],
                                                                           dim=2)
            predicted_ids = top_np_recommendation(guidance_candidate_ids, guidance_similarity_ratio,
                                                     confidence=torch.tensor(self.args.confidence),
                                                     threshold=0.8)
            return predicted_ids