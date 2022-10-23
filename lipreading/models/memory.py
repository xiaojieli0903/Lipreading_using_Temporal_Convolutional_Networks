import torch
import torch.nn.functional as F
from torch import nn

from lipreading.utils import calculate_loss


class Memory(nn.Module):
    def __init__(self,
                 radius=16.0,
                 n_slot=112,
                 n_head=8,
                 dim=512,
                 dim_query=64,
                 dim_mem=512,
                 dim_output=512,
                 diff_key_value=False,
                 choose_by_global=False,
                 no_norm=False,
                 use_hypotheses=False,
                 choose_type='cosine',
                 contrastive_hypo=False,
                 match_global=False,
                 use_kd=False,
                 value_adaptive=False,
                 loss_type='cosine'):
        super().__init__()
        self.diff_key_value = diff_key_value

        self.n_head = n_head
        self.n_slot = n_slot
        self.choose_by_global = choose_by_global
        self.no_norm = no_norm
        self.use_hypotheses = use_hypotheses
        assert choose_type in ['cosine', 'attention']
        self.choose_type = choose_type
        self.contrastive_hypo = contrastive_hypo
        self.dim_input = dim
        self.dim_query = dim_query
        self.dim_mem = dim_mem
        self.dim_output = dim_output
        self.match_global = match_global
        self.use_kd = use_kd
        self.value_adaptive = value_adaptive
        self.loss_type = loss_type

        self.key = nn.Parameter(torch.Tensor(n_head * n_slot, self.dim_query),
                                requires_grad=True)
        nn.init.normal_(self.key, 0, 0.5)
        self.value = nn.Parameter(torch.Tensor(n_slot, self.dim_mem),
                                  requires_grad=True)
        nn.init.normal_(self.value, 0, 0.5)

        if self.diff_key_value:
            if self.choose_by_global:
                self.global_proj_weight = nn.Linear(self.dim_input,
                                                    self.dim_mem)
            else:
                self.out_proj = nn.Linear(n_head * self.dim_mem,
                                          self.dim_output)
            if not self.no_norm:
                #self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(self.dim_output)
                self.norm3 = nn.LayerNorm(self.dim_output)
            self.v_up = nn.Linear(self.dim_mem, self.dim_output)
        else:
            if self.choose_by_global:
                self.global_proj_weight = nn.Linear(self.dim_input,
                                                    self.dim_mem)
                if self.choose_type == 'attention':
                    # global project to match local
                    self.global_key_proj_weight = nn.Linear(
                        dim, n_head * self.dim_mem)
            else:
                self.out_proj = nn.Linear(n_head * self.dim_mem,
                                          self.dim_output)
            if not self.no_norm:
                #self.norm1 = nn.LayerNorm(512)
                self.norm2 = nn.LayerNorm(self.dim_output)
                self.norm3 = nn.LayerNorm(self.dim_output)

        self.q_proj_weight = nn.Linear(self.dim_input, n_head * self.dim_query)
        self.v_proj_weight = nn.Linear(dim, self.dim_mem)
        if self.use_kd:
            self.knowledge_proj_weight = nn.Linear(self.dim_input,
                                                   self.dim_mem)

        # self.dropout = nn.Dropout(0.5)

        self.radius = radius
        self.softmax1 = nn.Softmax(2)
        self.softmax2 = nn.Softmax(1)

    def forward(self, query, value=None, f_global=None):
        # B, S, 512
        B, S, C = query.size()
        value = value.view(B * S, self.dim_input)  # BS,512
        f_target_recon, hypothesis_output = None, None
        recon_loss, contrastive_loss, hypo_contrastive_loss, \
        match_global_loss, kd_loss = torch.zeros(1).cuda(), \
                                     torch.zeros(1).cuda(), \
                                     torch.zeros(1).cuda(), \
                                     torch.zeros(1).cuda(), \
                                     torch.zeros(1).cuda()
        # n_head, n_slot * head_dim -->  # n_head, n_slot, head_dim
        key_normalized = F.normalize(self.key.view(self.n_head, self.n_slot,
                                                   self.dim_query),
                                     dim=2)
        # B*S, n_head * head_dim
        query_proj = self.q_proj_weight(query.view(B * S, self.dim_input))
        # BS, n_head, head_dim
        query_proj = query_proj.view(B * S, self.n_head, self.dim_query)
        query_proj = F.normalize(query_proj, dim=2)
        # BS, n_head, n_slot
        key_sim = torch.einsum('bhd,hsd->bhs', query_proj, key_normalized)
        # BS, n_head, n_slot
        key_address = self.softmax1(self.radius * key_sim)
        # ----visualize code
        # print(torch.max(key_address, dim=1))  # BS, predicts_times, n_slot

        # (BS, n_head, n_slot) * (n_slot , 512) --> BS, n_head, dim_mem
        if not self.value_adaptive:
            m_head_out = torch.matmul(key_address, self.value.detach())
        else:
            m_head_out = torch.matmul(key_address, self.value)

        if self.contrastive_hypo:
            m_head_out_normed = F.normalize(m_head_out, dim=2)
            out_sim = torch.einsum('bcd, bed->bce', m_head_out_normed,
                                   m_head_out_normed)
            # BS, n_head, n_head
            hypo_contrastive_loss = torch.abs(
                torch.eye(self.n_head).view(1, self.n_head, self.n_head).
                repeat(B * S, 1, 1).cuda() - out_sim).sum() / (B * S)
            # ----visualize code
            # sim_with_target = torch.einsum(
            #     'bhd, bd->bh', m_head_out_normed,
            #     F.normalize(value.view(B * S, -1), dim=-1))
            # torch.max(sim_with_target, dim=-1)
        if self.choose_by_global:
            # BS, n_head, dim_mem
            f_global_proj = self.global_proj_weight(f_global).view(
                B * S, self.dim_mem)
            f_global_norm = F.normalize(f_global_proj, dim=-1)
            if self.choose_type == 'attention':
                f_global_key_proj = self.global_key_proj_weight(f_global).view(
                    B * S, self.n_head, self.dim_mem)
                global_local_sim = torch.einsum(
                    'bhd,bd->bh', F.normalize(f_global_key_proj, dim=2),
                    f_global_norm)
            else:  # self.choose_type == 'cosine':
                global_local_sim = torch.einsum('bhd,bd->bh',
                                                F.normalize(m_head_out, dim=2),
                                                f_global_norm)
            if self.match_global:
                match_global_loss = calculate_loss(f_global_proj,
                                                   value,
                                                   loss_type=self.loss_type,
                                                   average_dim=0)
            hypothesis_address = self.softmax2(self.radius * global_local_sim)
            # if self.choose_type == 'max':
            #     max_idxs = torch.max(hypothesis_address, dim=1)[1]
            #     max_idxs_ = []
            #     for i in range(max_idxs.shape[0]):
            #         max_idxs_.append(max_idxs[i] + i * m_head_out.shape[1])
            #     attention_output = m_head_out.view(-1, C)[
            #         torch.tensor(max_idxs_).long()]
            # elif self.choose_type == 'cosine':
            # (BS , n_head) * (BS, n_head, dim_mem) -- > BS, dim_mem
            attention_output = torch.einsum('bh, bhd->bd', hypothesis_address,
                                            m_head_out)
            if self.use_hypotheses:
                # hypothesis_output = self.hypotheses_proj(
                #     m_head_out.view(B * S, self.dim_mem))
                hypothesis_output = m_head_out.detach()
        else:
            if not self.no_norm:
                attention_output = self.norm2(
                    self.out_proj(
                        m_head_out.view(B * S, self.n_head * self.dim_mem)))
            else:
                attention_output = self.out_proj(
                    m_head_out.view(B * S, self.n_head * self.dim_mem))

        f_predict = attention_output.view(B, S, self.dim_output)
        # f_predict = self.dropout(self.norm1(
        #     query + attention_output.view(B, S, -1)))
        if self.use_kd:
            # B*S, n_head * dim_mem
            query_kd_proj = self.knowledge_proj_weight(
                query.view(B * S, self.dim_input))
            # f_knowledge = torch.mean(torch.cat(
            #     (m_head_out, query.view(B*S, 1, self.dim_input)), dim=1),
            #     dim=1)

            # hypotheses aggregate: BS, n_head, dim_mem --> BS, dim_mem
            f_knowledge = torch.mean(m_head_out, dim=1)
            kd_loss = calculate_loss(query_kd_proj,
                                     f_knowledge.detach(),
                                     loss_type=self.loss_type,
                                     average_dim=0)
        # Update Value
        # n_slot, dim_mem
        value_proj = self.v_proj_weight(value)
        value_norm = F.normalize(self.value, dim=1)
        # BS, n_slot
        value_sim = F.linear(F.normalize(value_proj, dim=1), value_norm)
        value_address = self.softmax2(self.radius * value_sim)
        attention_recon = torch.matmul(value_address, self.value)
        # ----visualize code
        # n_slot * n_slot
        contrastive_loss = 0.01 * torch.abs(
            torch.eye(self.n_slot).cuda() -
            torch.matmul(value_norm, value_norm.transpose(0, 1))).sum()
        # print(torch.matmul(value_norm,
        #                    value_norm.transpose(0, 1))
        #       - torch.eye(self.n_slot).cuda())
        recon_loss = torch.abs(1.0 - F.cosine_similarity(
            attention_recon, value, 1)).sum() / (B * S)

        if self.diff_key_value:
            attention_recon = self.v_up(attention_recon)

        if not self.no_norm:
            attention_recon = self.norm3(attention_recon)

        f_target_recon = attention_recon.view(B, S, self.dim_output)
        # f_target_recon = self.dropout(self.norm1(
        #     query + attention_recon.view(B, S, -1)))

        return f_predict, f_target_recon, recon_loss, contrastive_loss, \
               hypothesis_output, hypo_contrastive_loss, match_global_loss, \
               kd_loss
