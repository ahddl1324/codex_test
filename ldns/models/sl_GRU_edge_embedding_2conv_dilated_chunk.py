import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

__all__ = ["GTSStructureLearnerGRU"]


class GTSStructureLearnerGRU(nn.Module):
    def __init__(
        self,
        observed_neurons: int,
        total_neurons: int,
        hidden_neurons: int,
        history: int,
        n_hid: int,
        HD_observed=None,
        HD_hidden=None,
        HD_total=None,
        edge_index: torch.Tensor = None,
        out_channel: int = 32,
        total_steps: int = 250,
        st_x: int = 1,
        receptive_field: Optional[int] = None,
        use_logvar: bool = False,
        logvar_bias_init: float = -4.0,
        embed_mode="learnable",
        emb_dim: int = 16,
        embed_ln: bool = False,
        embed_dropout: float = 0.0,
        gru_hidden: int = 32,
        gru_layers: int = 1,
        gru_dropout: float = 0.0,
        edge_chunk_num: int = 1,

        # ======================================================
        # New conv-equivalent settings
        # ======================================================
        local_conv_kernel: int = 80,
        local_conv_stride: int = 40,
        conv2_time_kernel: Optional[int] = None,
    ):
        super().__init__()

        self.observed_neurons = observed_neurons
        self.total_neurons = total_neurons
        self.hidden_neurons = hidden_neurons

        self.HD_observed = HD_observed
        self.HD_hidden = HD_hidden
        self.HD_total = HD_total

        self.n_hid = n_hid
        self.use_logvar = use_logvar
        self.emb_dim = emb_dim
        self.out_channel = out_channel
        self.st_x = st_x
        self.total_steps = total_steps
        self.edge_gru_hidden = gru_hidden
        self.edge_gru_layers = gru_layers

        self.edge_chunk_num = int(edge_chunk_num)
        if self.edge_chunk_num <= 0:
            raise ValueError(
                f"edge_chunk_num must be positive, got {self.edge_chunk_num}"
            )

        if edge_index is None:
            raise ValueError("edge_index must be provided.")

        self.register_buffer("edge_index", edge_index.long())

        # ======================================================
        # 1) Conv1d stride-1 + Dilated Conv2d block
        #
        # Goal:
        #   Original per-window operation:
        #
        #       x[t:t+200]
        #       -> Conv1D(kernel=80, stride=40, out=32)
        #       -> [32, 4]
        #       -> flatten / MLP
        #
        #   Equivalent full-sequence operation:
        #
        #       x[0:T]
        #       -> Conv1D(kernel=80, stride=1, out=32)
        #       -> [32, T-80+1]
        #       -> Conv2D(kernel=(32,4), dilation=(1,40))
        #       -> [n_hid, T-200+1]
        #
        # Example:
        #   total_steps = 400
        #   history / receptive_field = 200
        #   local_conv_kernel = 80
        #   local_conv_stride = 40
        #   conv2_time_kernel = 4
        #
        #   Conv1d:
        #       [B*C, 1, 400]
        #       -> [B*C, 32, 321]
        #
        #   Conv2d:
        #       [B*C, 1, 32, 321]
        #       kernel=(32,4), dilation=(1,40)
        #       effective RF on conv1 feature map:
        #           40 * (4 - 1) + 1 = 121
        #       original input RF:
        #           80 + 40 * (4 - 1) = 200
        #       -> [B*C, n_hid, 1, 201]
        #
        # output:
        #   [B, S, C, n_hid]
        #   S = (total_steps - receptive_field) // st_x + 1
        # ======================================================

        if receptive_field is None:
            receptive_field = history

        self.receptive_field = int(receptive_field)
        self.history = self.receptive_field

        self.conv1_kernel = int(local_conv_kernel)
        self.local_conv_stride = int(local_conv_stride)

        if self.conv1_kernel <= 0:
            raise ValueError(
                f"local_conv_kernel must be positive, got {self.conv1_kernel}"
            )

        if self.local_conv_stride <= 0:
            raise ValueError(
                f"local_conv_stride must be positive, got {self.local_conv_stride}"
            )

        if self.receptive_field < self.conv1_kernel:
            raise ValueError(
                f"receptive_field ({self.receptive_field}) must be >= "
                f"local_conv_kernel ({self.conv1_kernel})"
            )

        if total_steps < self.receptive_field:
            raise ValueError(
                f"total_steps ({total_steps}) must be >= "
                f"receptive_field ({self.receptive_field})"
            )

        # ------------------------------------------------------
        # conv2_time_kernel 자동 계산
        #
        # receptive_field = conv1_kernel + local_conv_stride * (conv2_time_kernel - 1)
        #
        # history=200, conv1_kernel=80, local_conv_stride=40:
        #   conv2_time_kernel = (200 - 80) / 40 + 1 = 4
        # ------------------------------------------------------
        if conv2_time_kernel is None:
            numerator = self.receptive_field - self.conv1_kernel

            if numerator % self.local_conv_stride != 0:
                raise ValueError(
                    "Cannot infer integer conv2_time_kernel. "
                    f"(receptive_field - local_conv_kernel) = {numerator} "
                    f"is not divisible by local_conv_stride = {self.local_conv_stride}. "
                    "Please pass conv2_time_kernel explicitly."
                )

            conv2_time_kernel = numerator // self.local_conv_stride + 1

        self.conv2_time_kernel = int(conv2_time_kernel)

        if self.conv2_time_kernel <= 0:
            raise ValueError(
                f"conv2_time_kernel must be positive, got {self.conv2_time_kernel}"
            )

        self.effective_receptive_field = (
            self.conv1_kernel
            + self.local_conv_stride * (self.conv2_time_kernel - 1)
        )

        if self.effective_receptive_field != self.receptive_field:
            raise ValueError(
                "Effective receptive field mismatch:\n"
                f"  requested receptive_field = {self.receptive_field}\n"
                f"  actual effective RF = conv1_kernel + dilation * (conv2_time_kernel - 1)\n"
                f"                      = {self.conv1_kernel} + "
                f"{self.local_conv_stride} * ({self.conv2_time_kernel} - 1)\n"
                f"                      = {self.effective_receptive_field}\n"
                "Adjust local_conv_kernel, local_conv_stride, or conv2_time_kernel."
            )

        self.convx_dim = (total_steps - self.receptive_field) // st_x + 1

        # ------------------------------------------------------
        # Conv1d:
        #   [B*C, 1, T]
        #   -> [B*C, out_channel, T - local_conv_kernel + 1]
        #
        # Important:
        #   stride=1로 전체 sequence feature map을 만든다.
        #   기존 window 내부 stride=40은 conv2d dilation으로 구현한다.
        # ------------------------------------------------------
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_channel,
            kernel_size=self.conv1_kernel,
            stride=1,
        )

        self.bn_conv1 = nn.BatchNorm1d(out_channel)

        # ------------------------------------------------------
        # Dilated Conv2d:
        #
        # input:
        #   [B*C, 1, out_channel, T1]
        #
        # kernel:
        #   height = out_channel
        #   width  = conv2_time_kernel
        #
        # dilation:
        #   time direction = local_conv_stride
        #
        # This samples conv1 features at:
        #   0, 40, 80, 120
        #
        # for local_conv_stride=40, conv2_time_kernel=4.
        # ------------------------------------------------------
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=n_hid,
            kernel_size=(out_channel, self.conv2_time_kernel),
            stride=(1, st_x),
            dilation=(1, self.local_conv_stride),
            padding=(0, 0),
        )

        self.bn_conv2 = nn.BatchNorm1d(n_hid)
        self.act = nn.ReLU()

        # ======================================================
        # 2) Positional Embedding
        # ======================================================
        self.embed_mode = embed_mode

        if self.embed_mode == "learnable":
            self.E = nn.Parameter(
                torch.randn(observed_neurons, emb_dim)
            )

        elif self.embed_mode == "fixed":
            self.register_buffer(
                "E",
                torch.randn(observed_neurons, emb_dim)
            )

        elif self.embed_mode == "none":
            self.E = None

        else:
            raise ValueError(
                f"Unknown embed_mode: {self.embed_mode}. "
                f"Choose from ['none', 'fixed', 'learnable']"
            )

        self.embed_ln = (
            nn.LayerNorm(emb_dim)
            if (embed_ln and self.E is not None)
            else None
        )

        self.embed_dropout = (
            nn.Dropout(embed_dropout)
            if (embed_dropout > 0 and self.E is not None)
            else None
        )

        # ======================================================
        # 3) Edge pair 2-layer MLP -> 32 dim
        # ======================================================
        node_feat_dim = n_hid + (emb_dim if self.E is not None else 0)
        pair_feat_dim = 2 * node_feat_dim

        self.edge_pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # ======================================================
        # 4) Edge-level GRU
        #
        # edge_feat:
        #   [B*E, S, 32]
        #
        # chunking:
        #   [B*E] 축을 edge_chunk_num개로 나눠 GRU 수행
        # ======================================================
        self.edge_gru = nn.GRU(
            input_size=32,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )

        # ======================================================
        # 5) 3-layer MLP for connectivity inference
        # ======================================================
        self.edge_out_mlp = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        if self.use_logvar:
            self.edge_logvar_mlp = nn.Sequential(
                nn.Linear(gru_hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

            with torch.no_grad():
                self.edge_logvar_mlp[-1].bias.fill_(logvar_bias_init)

        self._init_weights()

    # ------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------
    @torch.no_grad()
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    # ------------------------------------------------------------
    @staticmethod
    def make_chunk_slices(total_seq: int, n_chunks: int):
        """
        total_seq를 n_chunks개로 최대한 균등하게 나눔.

        Used for:
            edge_feat: [B*E, S, D]

        If n_chunks > total_seq, clamp to total_seq.
        """
        if n_chunks <= 0:
            raise ValueError(f"n_chunks must be positive, got {n_chunks}")

        if n_chunks > total_seq:
            n_chunks = total_seq

        base = total_seq // n_chunks
        rem = total_seq % n_chunks

        slices = []
        start = 0

        for k in range(n_chunks):
            chunk_size = base + 1 if k < rem else base
            end = start + chunk_size
            slices.append((start, end))
            start = end

        assert slices[-1][1] == total_seq

        return slices

    # ------------------------------------------------------------
    def run_edge_gru_chunked(
        self,
        edge_feat: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        edge_feat:
            [B*E, S, 32]

        h0:
            None or [edge_gru_layers, B*E, edge_gru_hidden]

        returns:
            edge_h_seq:
                [B*E, S, edge_gru_hidden]

            h_last:
                [edge_gru_layers, B*E, edge_gru_hidden]

        Chunking is along [B*E] axis.
        """
        if edge_feat.dim() != 3:
            raise ValueError(
                f"run_edge_gru_chunked expects [B*E,S,D], "
                f"got {tuple(edge_feat.shape)}"
            )

        total_seq = edge_feat.size(0)

        if h0 is not None:
            if h0.dim() != 3:
                raise ValueError(
                    f"h0 must be [layers,B*E,H], got {tuple(h0.shape)}"
                )

            if h0.size(0) != self.edge_gru_layers:
                raise ValueError(
                    f"h0 layers mismatch: expected {self.edge_gru_layers}, "
                    f"got {h0.size(0)}"
                )

            if h0.size(1) != total_seq:
                raise ValueError(
                    f"h0 B*E mismatch: expected {total_seq}, got {h0.size(1)}"
                )

            if h0.size(2) != self.edge_gru_hidden:
                raise ValueError(
                    f"h0 hidden mismatch: expected {self.edge_gru_hidden}, "
                    f"got {h0.size(2)}"
                )

        # no chunk
        if self.edge_chunk_num <= 1:
            return self.edge_gru(edge_feat, h0)

        slices = self.make_chunk_slices(total_seq, self.edge_chunk_num)

        out_parts = []
        h_parts = []

        for e0, e1 in slices:
            edge_part = edge_feat[e0:e1]
            # [chunk, S, 32]

            if h0 is None:
                h0_part = None
            else:
                h0_part = h0[:, e0:e1, :].contiguous()
                # [layers, chunk, H]

            out_part, h_part = self.edge_gru(edge_part, h0_part)

            out_parts.append(out_part)
            h_parts.append(h_part)

        edge_h_seq = torch.cat(out_parts, dim=0)
        # [B*E, S, H]

        h_last = torch.cat(h_parts, dim=1)
        # [layers, B*E, H]

        return edge_h_seq, h_last

    # ------------------------------------------------------------
    def get_positional_embedding_sequence(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """
        return:
            None or [B, S, C, emb_dim]
        """
        if self.E is None:
            return None

        E = self.E

        if device is not None or dtype is not None:
            E = E.to(
                device=device if device is not None else E.device,
                dtype=dtype if dtype is not None else E.dtype,
            )

        if self.embed_ln is not None:
            E = self.embed_ln(E)

        if self.embed_dropout is not None:
            E = self.embed_dropout(E)

        C = self.observed_neurons

        E_seq = E.view(1, 1, C, self.emb_dim)
        E_seq = E_seq.expand(batch_size, seq_len, C, self.emb_dim)

        return E_seq

    # ------------------------------------------------------------
    def encode_nodes_sequence(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        inputs:
            [C, T_long] or [B, C, T_long]

        returns:
            z_seq:
                [S, C, n_hid] or [B, S, C, n_hid]
        """
        squeeze_batch = False

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
            squeeze_batch = True

        if inputs.dim() != 3:
            raise ValueError(
                f"encode_nodes_sequence expects [C,T] or [B,C,T], "
                f"got {tuple(inputs.shape)}"
            )

        B, C, T = inputs.shape

        if C != self.observed_neurons:
            raise ValueError(
                f"Expected C={self.observed_neurons}, got C={C}"
            )

        if T < self.receptive_field:
            raise ValueError(
                f"Input T ({T}) must be >= receptive_field ({self.receptive_field})"
            )

        # ======================================================
        # 1) reshape
        # [B, C, T] -> [B*C, 1, T]
        # ======================================================
        x = inputs.reshape(B * C, 1, T)

        # ======================================================
        # 2) Conv1d stride=1
        #
        # Example:
        #   T=400, local_conv_kernel=80
        #   [B*C, 1, 400]
        #   -> [B*C, 32, 321]
        # ======================================================
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn_conv1(x)

        # ======================================================
        # 3) Dilated Conv2d
        #
        # [B*C, 32, T1]
        # -> [B*C, 1, 32, T1]
        #
        # kernel=(32,4), dilation=(1,40)
        #
        # Example:
        #   T1=321
        #   effective width on T1 = 40*(4-1)+1 = 121
        #   output S = 321 - 121 + 1 = 201
        # ======================================================
        x = x.unsqueeze(1)

        x = self.conv2d(x)
        x = self.act(x)

        # [B*C, n_hid, 1, S] -> [B*C, n_hid, S]
        x = x.squeeze(2)
        x = self.bn_conv2(x)

        # ======================================================
        # 4) reshape to node sequence
        #
        # [B*C, n_hid, S]
        # -> [B*C, S, n_hid]
        # -> [B, C, S, n_hid]
        # -> [B, S, C, n_hid]
        # ======================================================
        z = x.transpose(1, 2).contiguous()

        S = z.size(1)

        z = z.reshape(B, C, S, self.n_hid)
        z = z.permute(0, 2, 1, 3).contiguous()

        if squeeze_batch:
            z = z.squeeze(0)

        return z

    # ------------------------------------------------------------
    def edge_params_from_z_sequence(
        self,
        z_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
    ]:
        """
        z_seq:
            [S, C, n_hid] or [B, S, C, n_hid]

        h0:
            optional edge-GRU hidden
            shape:
                [edge_gru_layers, B*E, edge_gru_hidden]

        returns:
            mu:
                [S, E, 1] or [B, S, E, 1]

            logvar:
                None or same shape as mu

            h_last:
                optional edge-GRU final hidden
                [edge_gru_layers, B*E, edge_gru_hidden]
        """
        squeeze_batch = False

        if z_seq.dim() == 3:
            z_seq = z_seq.unsqueeze(0)
            squeeze_batch = True

        if z_seq.dim() != 4:
            raise ValueError(
                f"edge_params_from_z_sequence expects [S,C,F] or [B,S,C,F], "
                f"got {tuple(z_seq.shape)}"
            )

        B, S, C, F = z_seq.shape

        if C != self.observed_neurons:
            raise ValueError(
                f"Expected C={self.observed_neurons}, got C={C}"
            )

        if F != self.n_hid:
            raise ValueError(
                f"Expected node feature dim n_hid={self.n_hid}, got F={F}"
            )

        src = self.edge_index[0].long()
        dst = self.edge_index[1].long()
        E_num = src.numel()

        # ======================================================
        # 1) Node feature gather
        # zi, zj:
        #   [B, S, E, n_hid]
        # ======================================================
        zi = z_seq[:, :, src, :]
        zj = z_seq[:, :, dst, :]

        node_pair = torch.cat([zi, zj], dim=-1)

        # ======================================================
        # 2) PE concat
        # ======================================================
        if self.E is not None:
            E_seq = self.get_positional_embedding_sequence(
                batch_size=B,
                seq_len=S,
                device=z_seq.device,
                dtype=z_seq.dtype,
            )

            Ei = E_seq[:, :, src, :]
            Ej = E_seq[:, :, dst, :]

            pe_pair = torch.cat([Ei, Ej], dim=-1)

            pair = torch.cat([node_pair, pe_pair], dim=-1)
        else:
            pair = node_pair

        # ======================================================
        # 3) Edge pair MLP
        # [B, S, E, pair_feat_dim] -> [B, S, E, 32]
        # ======================================================
        edge_feat = self.edge_pair_mlp(pair)

        # ======================================================
        # 4) Edge-level GRU
        #
        # [B, S, E, 32]
        # -> [B, E, S, 32]
        # -> [B*E, S, 32]
        # -> chunked GRU over S dimension
        # ======================================================
        edge_feat = edge_feat.permute(0, 2, 1, 3).contiguous()
        edge_feat = edge_feat.reshape(B * E_num, S, 32)

        edge_h_seq, h_last = self.run_edge_gru_chunked(edge_feat, h0)

        # [B*E, S, H] -> [B, E, S, H] -> [B, S, E, H]
        edge_h_seq = edge_h_seq.reshape(B, E_num, S, self.edge_gru_hidden)
        edge_h_seq = edge_h_seq.permute(0, 2, 1, 3).contiguous()

        # ======================================================
        # 5) Output MLP
        # [B, S, E, H] -> [B, S, E, 1]
        # ======================================================
        mu = self.edge_out_mlp(edge_h_seq)

        logvar = None

        if self.use_logvar:
            logvar = self.edge_logvar_mlp(edge_h_seq)

        if squeeze_batch:
            mu = mu.squeeze(0)
            if logvar is not None:
                logvar = logvar.squeeze(0)

        if return_hidden:
            return mu, logvar, h_last

        return mu, logvar

    # ------------------------------------------------------------
    def forward(
        self,
        inputs: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_hidden: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        inputs:
            [C, T_long] or [B, C, T_long]

        output:
            if use_logvar:
                mu, logvar
            else:
                mu

        mu shape:
            [S, E, 1] or [B, S, E, 1]

        if return_features:
            append z_seq

        if return_hidden:
            append h_last
        """
        z_seq = self.encode_nodes_sequence(inputs)

        if return_hidden:
            mu, logvar, h_last = self.edge_params_from_z_sequence(
                z_seq,
                h0=h0,
                return_hidden=True,
            )
        else:
            mu, logvar = self.edge_params_from_z_sequence(
                z_seq,
                h0=h0,
                return_hidden=False,
            )
            h_last = None

        outputs = []

        if self.use_logvar:
            outputs = [mu, logvar]
        else:
            outputs = [mu]

        if return_features:
            outputs.append(z_seq)

        if return_hidden:
            outputs.append(h_last)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)