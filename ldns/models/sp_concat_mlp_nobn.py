import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class sp_ver2(nn.Module):
    """
    입력:
      inputs: [H, B, C_obs]   ← 로더 출력과 동일 (TH/no_external에서 각도 미사용)
      angs  : [B, T] 또는 dummy (TH에서는 무시)
      z     : [B, E, 1]       (edge-wise gate; SL에서 [E,1]을 batch로 expand해서 전달)
    출력:
      preds: [C_obs, B, pred_steps, 1]  (log-rate)
    """
    def __init__(self, observed_neurons, total_neurons, hidden_neurons, history, n_hid,
                 HD_observed, HD_hidden, HD_total, edge_idx,
                 session='TH', trial=False, msg_hop=1, do_prob=0., shared_conv=None, residual_flag=False):
        super().__init__()
        self.hidden_dim = n_hid
        self.out_channel = n_hid
        self.do_prob = do_prob
        self.observed_neurons = observed_neurons
        self.total_neurons = total_neurons
        self.hidden_neurons = hidden_neurons
        self.history = history
        self.edge_index = edge_idx
        self.HD_total = HD_total
        self.HD_observed = HD_observed
        self.HD_hidden = HD_hidden
        self.session = session
        self.trial = trial
        self.residual_flag = residual_flag

        # TH에서는 각도 입력 없음
        if self.session not in ('no_external', 'TH'):
            self.ang_bias = nn.Parameter(torch.zeros(1))

        # Conv1d(1 -> n_hid, kernel_size=history, stride=1) 권장
        # (shared_conv를 밖에서 같은 설정으로 만들어 넘겨줘)
        self.shared_conv = shared_conv
        #self.BN = nn.BatchNorm1d(n_hid)
        self.f_x = nn.Sequential(
            shared_conv,
            nn.ReLU(),
            #nn.BatchNorm1d(n_hid),
            nn.Identity()
        )
        #self.register_buffer("edge_index", edge_idx)

        # 각도 사용 안 하는 TH/no_external 경로
        # if self.session != 'no_external' and self.session != 'TH':
        #     self.f_theta = nn.Sequential(
        #         nn.Linear(1, n_hid), nn.ReLU(),
        #         nn.Linear(n_hid, n_hid), nn.ReLU()
        #     )
        #     self.h_out = nn.Sequential(nn.Linear(2*n_hid, n_hid))
        # else:
        #     #self.h_out = nn.Sequential(nn.Linear(n_hid, n_hid))

        self.f_out = nn.Sequential(nn.ReLU(), nn.Linear(n_hid, 1))

        if self.trial:
            self.generate_hidden = mpgnn2(total_neurons, edge_idx, 2, n_hid, session, trial)
            self.hidden_z = nn.Parameter(torch.ones(total_neurons*(total_neurons-1), 1),
                                         requires_grad=False)

        self.spike_prediction = mpgnn2(total_neurons, edge_idx, msg_hop, n_hid, session, trial)

    def forward(self, inputs, angs, z, pred_steps=1):
        """
        inputs: [H, B, C_obs]
        angs  : [B, T] (TH/no_external에서는 무시)
        z     : [B, E, 1]  또는 [E,1] (이 경우 배치로 확장)
        """
        H, B, Cobs = inputs.shape
        assert H == self.history and Cobs == self.observed_neurons, \
            f"Expected inputs[{self.history}, B, {self.observed_neurons}], got {inputs.shape}"

        # ---- z를 [B,E,1]로 보정 ----
        if z.dim() == 2:                 # [E,1]
            z = z.unsqueeze(0).expand(B, -1, -1).contiguous()
        elif z.dim() == 3 and z.size(0) == 1:  # [1,E,1]
            z = z.expand(B, -1, -1).contiguous()
        assert z.dim() == 3 and z.size(0) == B and z.size(2) == 1, f"z must be [B,E,1], got {z.shape}"

        pred_all = []

        # [H,B,C] -> [C,B,H]
        x_seq = inputs.permute(2, 1, 0).contiguous()  # [C_obs, B, H]

        for step in range(pred_steps):
            if step == 0:
                y_hat = x_seq                          # [C_obs, B, H]
            else:
                lam = torch.exp(pred_prev)             # [C_obs, B, 1]
                samp = torch.poisson(lam)              # [C_obs, B, 1]
                y_hat = torch.cat((y_hat[:, :, 1:], samp), dim=-1)

            # Conv1d: [C_obs*B, 1, H] -> [C_obs*B, n_hid, 1] -> [C_obs, B, d]
            y_hat_ = y_hat.reshape(-1, 1, H)
            h_emb = self.f_x(y_hat_)
            h_emb = h_emb.reshape(Cobs, B, self.hidden_dim)

            # 메시지 패싱 (배치별 z 사용)
            h_emb = self.spike_prediction(h_emb, z)    # [C_total, B, d]

            # 관측 뉴런만
            h_obs = h_emb[:Cobs, :, :]                 # [C_obs, B, d]
            #h_obs = self.h_out(h_obs)                  # [C_obs, B, d]
            pred_residual = self.f_out(h_obs)        # 항상 residual 값으로 계산됨

            if self.residual_flag:
                # 이전 프레임 값에 residual을 더해서 다음 프레임 추론
                pred = pred_residual + y_hat[:, :, -1:]   # y_hat 마지막 프레임이 직전값
            else:
                # 절대값을 직접 예측
                pred = pred_residual                  # [C_obs, B, 1] (log-rate)
            pred_all.append(pred)
            pred_prev = pred

        preds = torch.stack(pred_all, dim=2)           # [C_obs, B, pred_steps, 1]
        return preds


class mpgnn2(MessagePassing):
    def __init__(self, total_neurons, edge_index, msg_hop, n_hid, session, trial):
        super(mpgnn2, self).__init__(aggr='add')
        self.total_neurons = total_neurons
        self.edge_index = edge_index
        self.msg_hop = msg_hop
        self.hidden_dim = n_hid
        self.session = session
        self.trial = trial

        if self.session != 'no_external' and self.session != 'TH':
            hidden_dim = 2 * n_hid

            # concat(x_i, x_j): 입력 차원 = 2 * hidden_dim = 4 * n_hid
            self.psi_h = nn.Sequential(
                nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.gru_h = nn.GRUCell(2 * hidden_dim, hidden_dim)

        else:
            hidden_dim = n_hid

            # concat(x_i, x_j): 입력 차원 = 2 * hidden_dim = 2 * n_hid
            self.psi_h = nn.Sequential(
                nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.gru_h = nn.GRUCell(2 * hidden_dim, hidden_dim)

        # self.register_buffer("edge_index", edge_index)

    def forward(self, h_emb, z):  # h_emb: [total_neurons, B, H]
        h_0 = h_emb
        for _ in range(self.msg_hop):
            h_emb = h_emb.reshape(self.total_neurons, -1)
            h_emb = self.propagate(
                edge_index=self.edge_index,
                x=h_emb,
                z=z,
                h_0=h_0,
                h_emb=h_emb
            )
        return h_emb

    def message(self, x_i, x_j, z):
        if self.session != 'no_external' and self.session != 'TH':
            hidden_dim = 2 * self.hidden_dim
        else:
            hidden_dim = self.hidden_dim

        # x_i, x_j: [E, B*H]
        E = x_i.size(0)
        B = z.size(0)
        H = hidden_dim

        # [E, B*H] -> [E, B, H]
        x_i = x_i.view(E, B, H)
        x_j = x_j.view(E, B, H)

        # difference 대신 concat
        msg_input = torch.cat([x_j, x_i], dim=-1)   # [E, B, 2H]

        # psi_h 적용
        msg = self.psi_h(msg_input.reshape(E * B, 2 * H)).reshape(E, B, H)  # [E, B, H]

        # z: [B, E, 1] -> [E, B, 1]
        assert z.dim() == 3 and z.size(0) == B and z.size(2) == 1, \
            f"z must be [B,E,1], got {tuple(z.shape)}"
        z_e_b_1 = z.permute(1, 0, 2)   # [E, B, 1]

        msg = msg * z_e_b_1            # [E, B, H]

        # [E, B, H] -> [E, B*H]
        msg = msg.reshape(E, B * H)
        return msg

    def update(self, aggr_msg, h_0, h_emb):
        if self.session != 'no_external' and self.session != 'TH':
            hidden_dim = 2 * self.hidden_dim
        else:
            hidden_dim = self.hidden_dim

        aggr_msg = aggr_msg.reshape(self.total_neurons, -1, hidden_dim)
        aggr_msg = torch.cat((aggr_msg, h_0), dim=-1)   # [total_neurons, B, 2H]
        aggr_msg = aggr_msg.reshape(-1, 2 * hidden_dim)

        h_emb = h_emb.reshape(-1, hidden_dim)
        h_emb = self.gru_h(aggr_msg, h_emb)
        h_emb = h_emb.reshape(self.total_neurons, -1, hidden_dim)

        return h_emb