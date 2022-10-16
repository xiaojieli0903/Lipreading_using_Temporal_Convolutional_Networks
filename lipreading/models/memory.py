import torch
import torch.nn.functional as F
from torch import nn


class Memory(nn.Module):
    def __init__(self,
                 radius=16.0,
                 n_slot=112,
                 n_head=8,
                 dim=512,
                 diff_key_value=False,
                 fix_memory=False,
                 choose_by_context=False,
                 no_norm=False):
        super().__init__()
        self.diff_key_value = diff_key_value

        self.n_head = n_head
        self.n_slot = n_slot
        self.fix_memory = fix_memory
        self.choose_by_context = choose_by_context
        self.no_norm = no_norm

        self.key = nn.Parameter(torch.Tensor(int(n_head * n_slot),
                                             int(512 / n_head)),
                                requires_grad=True)
        nn.init.normal_(self.key, 0, 0.5)
        self.value = nn.Parameter(torch.Tensor(n_slot, 512),
                                  requires_grad=True)
        nn.init.normal_(self.value, 0, 0.5)

        if self.diff_key_value:
            if self.choose_by_context:
                self.context_proj_weight = nn.Linear(dim, 512)
            else:
                self.out_proj = nn.Linear(512 * n_head, dim)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.norm3 = nn.LayerNorm(dim)
            self.v_up = nn.Linear(512, dim)
        else:
            if self.choose_by_context:
                self.context_proj_weight = nn.Linear(dim, 512)
            else:
                self.out_proj = nn.Linear(512 * n_head, 512)
            self.norm1 = nn.LayerNorm(512)
            self.norm2 = nn.LayerNorm(512)
            self.norm3 = nn.LayerNorm(512)

        self.q_proj_weight = nn.Linear(dim, 512)
        self.v_proj_weight = nn.Linear(512, 512)

        self.dropout = nn.Dropout(0.5)

        self.radius = radius
        self.softmax1 = nn.Softmax(2)
        self.softmax2 = nn.Softmax(1)

    def forward(self, query, value=None, f_exclude_predicts=None, inference=False):
        # B, S, 512
        B, S, C = query.size()
        f_target_recon, recon_loss, contrastive_loss = None, torch.zeros(1).cuda(), torch.zeros(1).cuda()

        key_normalized = F.normalize(self.key.view(self.n_head, self.n_slot, -1), dim=2)  # n_head, n_slot, head_dim
        query_proj = self.q_proj_weight(query.view(B * S, -1))  # B*S, n_head * head_dim
        query_proj = query_proj.view(B * S, self.n_head, -1)  # BS, n_head, head_dim
        query_proj = F.normalize(query_proj, dim=2)

        key_sim = torch.einsum('bhd,hsd->bhs', query_proj, key_normalized)  # BS, n_head, n_slot
        key_address = self.softmax1(self.radius * key_sim)  # BS, n_head, n_slot
        # (BS, n_head, n_slot) * (n_slot , 512) --> BS, n_head, 512
        m_head_out = torch.matmul(key_address, self.value.detach())

        if self.choose_by_context:
            m_head_out = m_head_out.view(B * S, self.n_head, -1)  # BS, n_head, head_dim
            f_exclude_predicts_proj = self.context_proj_weight(f_exclude_predicts.detach()).view(B * S, -1)
            f_exclude_predicts_norm = F.normalize(f_exclude_predicts_proj, dim=-1)  # BS, head_dim
            context_hypothesis_sim = torch.einsum('bhd,bd->bh', F.normalize(m_head_out, dim=2), f_exclude_predicts_norm)
            hypothesis_address = self.softmax2(self.radius * context_hypothesis_sim)  # BS , n_head
            # (BS , n_head) * (BS, n_head, head_dim)
            attention_output = torch.einsum('bh, bhd->bd', hypothesis_address, m_head_out)  # BS, head_dim
        else:
            m_head_out = m_head_out.view(B * S, -1)  # BS, n_head*512
            if self.no_norm:
                attention_output = self.norm2(self.out_proj(m_head_out))  # BS, 512
            else:
                attention_output = self.out_proj(m_head_out)  # BS, 512

        if self.fix_memory:
            f_predict = attention_output.view(B, S, -1)
        else:
            f_predict = self.dropout(self.norm1(query + attention_output.view(B, S, -1)))

        # Update
        if not inference:
            value = value.view(B * S, -1)  # BS,512
            value_proj = self.v_proj_weight(value.detach())
            value_norm = F.normalize(self.value, dim=1)  # n_slot,512
            value_sim = F.linear(F.normalize(value_proj, dim=1), value_norm)  # BS, n_slot
            value_address = self.softmax2(self.radius * value_sim)

            attention_recon = torch.matmul(value_address, self.value)  # BS,512

            contrastive_loss = torch.abs(
                torch.eye(self.n_slot).cuda() -
                torch.matmul(value_norm, value_norm.transpose(0, 1))).sum() * 0.01  # n_slot, n_slot

            recon_loss = torch.abs(1.0 - F.cosine_similarity(attention_recon, value.detach(), 1)).sum() / (B * S)

            if self.diff_key_value:
                attention_recon = self.v_up(attention_recon)

            attention_recon = self.norm3(attention_recon)
            if self.fix_memory:
                f_target_recon = attention_recon.view(B, S, -1)
            else:
                f_target_recon = self.dropout(self.norm1(query + attention_recon.view(B, S, -1)))

        return f_predict, f_target_recon, recon_loss, contrastive_loss
