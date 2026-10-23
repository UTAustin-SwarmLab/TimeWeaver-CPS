import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


from timeseries_synthesis.utils.basic_utils import (
    get_denoiser_config,
    get_dataset_config,
    OKBLUE,
    ENDC,
)

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiscreteConditionEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, initial_projection_dim, projection_dim, dropout):
        super().__init__()
        self.projection1 = torch.nn.Linear(embedding_dim, initial_projection_dim)
        self.projection2 = torch.nn.Linear(initial_projection_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(projection_dim, projection_dim)
        self.fc2 = torch.nn.Linear(projection_dim, projection_dim)
        self.fc3 = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.layer_norm1 = torch.nn.LayerNorm(projection_dim)
        self.layer_norm2 = torch.nn.LayerNorm(projection_dim)
        self.layer_norm3 = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection1(inp)
        projected = self.gelu(projected)
        projected = self.projection2(projected)
        x = self.gelu(projected)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = x + projected
        x = self.layer_norm1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + projected
        x = self.layer_norm2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = x + projected
        x = self.layer_norm3(x)
        return x


class ConditionEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(projection_dim, projection_dim)
        self.fc2 = torch.nn.Linear(projection_dim, projection_dim)
        self.fc3 = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.layer_norm1 = torch.nn.LayerNorm(projection_dim)
        self.layer_norm2 = torch.nn.LayerNorm(projection_dim)
        self.layer_norm3 = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection(inp)
        x = self.gelu(projected)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = x + projected
        x = self.layer_norm1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + projected
        x = self.layer_norm2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = x + projected
        x = self.layer_norm3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_info, diffusion_emb, cond_in):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        # print(x.shape, diffusion_emb.shape, cond_in.shape)
        y = x + diffusion_emb + cond_in  # (B,channel,K*L)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)
        # print(y.shape)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.side_projection(side_info)  # (B,2*channel,K*L)
        # print(y.shape, side_info.shape)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "diffusion_embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.diffusion_embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class CSDITSDenoiser_v1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser_config = get_denoiser_config(config=self.config)
        # self.diffusion_config = self.config.diffusion_config
        self.dataset_config = get_dataset_config(config=self.config)
        self.device = self.config.device
        # self.cltsp_config = get_cltsp_config(config=self.config)

        # for each timestep in the timeseries, we have a positional embedding
        self.pos_embedding_dim = self.denoiser_config.positional_embedding_dim  # 128
        self.channels = self.denoiser_config.channels  # 512
        self.num_features = self.dataset_config.num_channels  # K
        
        T = 200
        beta_0 = 0.0001
        beta_T = 0.1
        self.num_steps = T  # 50 steps
        self.diffusion_hyperparameters = self.calc_diffusion_hyperparams(
            T=T,
            beta_0=beta_0,
            beta_T=beta_T,
        )

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=T,
            embedding_dim=self.channels,
        )
        self.channel_embedding = torch.nn.Embedding(
            num_embeddings=self.num_features,
            embedding_dim=self.denoiser_config.channel_embedding_dim,
        )
        
        num_discrete_labels = self.dataset_config.num_discrete_labels
        num_discrete_conditions = self.dataset_config.num_discrete_conditions
        num_continuous_labels = self.dataset_config.num_continuous_labels

        self.discrete_condition_encoder_exists = (
            True if num_discrete_labels > 0 else False
        )
        self.continuous_condition_encoder_exists = (
            True if num_continuous_labels > 0 else False
        )
        self.combined_condition_encoder_exists = (
            self.discrete_condition_encoder_exists
            and self.continuous_condition_encoder_exists
        )

        if self.discrete_condition_encoder_exists:
            self.discrete_condition_encoder = DiscreteConditionEncoder(
                embedding_dim=num_discrete_labels,
                initial_projection_dim=int(
                    num_discrete_conditions
                    * self.dataset_config.discrete_condition_embedding_dim
                ),
                projection_dim=self.channels,
                dropout=0.1,
            )
            print(OKBLUE + "Discrete condition encoder exists" + ENDC)
            print(
                OKBLUE
                + "Discrete condition encoder input size = %d" % num_discrete_labels
                + ENDC
            )

        if self.continuous_condition_encoder_exists:
            self.continuous_condition_encoder = ConditionEncoder(
                embedding_dim=num_continuous_labels,
                projection_dim=self.channels,
                dropout=0.1,
            )
            print(OKBLUE + "Continuous condition encoder exists" + ENDC)
            print(
                OKBLUE
                + "Continuous condition encoder input size = %d" % num_continuous_labels
                + ENDC
            )

        if self.combined_condition_encoder_exists:
            self.combined_condition_encoder = ConditionEncoder(
                embedding_dim=self.channels * 2,
                projection_dim=self.channels,
                dropout=0.1,
            )
            print(OKBLUE + "Combined condition encoder exists" + ENDC)

        self.input_projection = Conv1d_with_init(1, self.channels, 1)  # 1 to 256

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=self.pos_embedding_dim
                    + self.denoiser_config.channel_embedding_dim,  # 128
                    channels=self.channels,  # 256
                    diffusion_embedding_dim=self.channels,  # 256
                    nheads=self.denoiser_config.n_heads,  # 16
                )
                for _ in range(self.denoiser_config.n_layers)
            ] 
        )

    def calc_diffusion_hyperparams(self, T, beta_0, beta_T):
        Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
        Alpha = 1 - Beta
        Alpha_bar = Alpha + 0
        Beta_tilde = Beta + 0
        for t in range(1, T):
            Alpha_bar[t] *= Alpha_bar[t - 1]
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
        Sigma = torch.sqrt(Beta_tilde)

        Beta = Beta.to(self.device)
        Alpha = Alpha.to(self.device)
        Alpha_bar = Alpha_bar.to(self.device)
        Sigma = Sigma.to(self.device)

        _dh = {}
        _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
            T,
            Beta,
            Alpha,
            Alpha_bar,
            Sigma,
        )
        diffusion_hyperparams = _dh
        return diffusion_hyperparams

    def position_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, time_points):
        B = time_points.shape[0]
        L = time_points.shape[1]
        time_embed = self.position_embedding(
            time_points, self.pos_embedding_dim
        )  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).repeat(
            1, 1, self.num_features, 1
        )  # (B, L, K, emb)
        feature_embed = self.channel_embedding(
            torch.arange(self.num_features).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, emb, K, L  )
        return side_info.to(self.device)

    def prepare_training_input(self, train_batch):
        # sample
        sample = train_batch["timeseries_full"].float().to(self.device)

        assert sample.shape[1] == self.num_features

        # discrete and continuous condition input
        discrete_label_embedding = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )
        continuous_label_embedding = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )

        # diffusion step
        _dh = self.diffusion_hyperparameters
        Alpha_bar = _dh["Alpha_bar"]
        batch_size = sample.shape[0]
        t = torch.randint(
            0,
            self.num_steps,
            (batch_size,),
        )

        # noise and noisy data
        current_alpha_bar = Alpha_bar[t].unsqueeze(1).unsqueeze(1).to(self.device)
        print(self.alpha[t], Alpha_bar[t])
        noise = torch.randn_like(sample).float().to(self.device)
        noisy_data = (current_alpha_bar**0.5) * sample + (
            1.0 - current_alpha_bar
        ) ** 0.5 * noise

        # print(side_info.shape)

        denoiser_input = {
            "noisy_sample": noisy_data,
            "noise": noise,
            "sample": sample,
            "discrete_cond_input": discrete_label_embedding,
            "continuous_cond_input": continuous_label_embedding,
            "diffusion_step": t,
        }

        return denoiser_input

    def forward(self, denoiser_input):
        noisy_input = denoiser_input["noisy_sample"]  # (B, K, L)
        discrete_cond_input = denoiser_input["discrete_cond_input"]
        continuous_cond_input = denoiser_input["continuous_cond_input"]
        diffusion_step = denoiser_input["diffusion_step"]
        # convert diffusion_step to a long tensor
        diffusion_step = diffusion_step.long()
        # print(
        #     noisy_input.shape,
        #     discrete_cond_input.shape,
        #     continuous_cond_input.shape,
        #     diffusion_step.shape,
        # )

        B = noisy_input.shape[0]  # B
        K = noisy_input.shape[1]  # K
        L = noisy_input.shape[2]  # L

        tp = torch.arange(L).unsqueeze(0).repeat(B, 1).float().to(self.device)
        side_info = self.get_side_info(tp)

        if self.discrete_condition_encoder_exists:
            discrete_cond_in = self.discrete_condition_encoder(discrete_cond_input)
        if self.continuous_condition_encoder_exists:
            continuous_cond_in = self.continuous_condition_encoder(
                continuous_cond_input
            )

        if self.combined_condition_encoder_exists:
            cond_in = torch.cat([discrete_cond_in, continuous_cond_in], dim=-1)
            cond_in = self.combined_condition_encoder(cond_in)
        elif self.discrete_condition_encoder_exists:
            cond_in = discrete_cond_in
        elif self.continuous_condition_encoder_exists:
            cond_in = continuous_cond_in
        cond_in = cond_in.unsqueeze(-1).repeat(1, 1, K * L)  # (B,1,K*L)

        x = noisy_input.reshape(B, K * L).unsqueeze(1)  # (B,1,K*L), scales the input by different self.channels 
        x = self.input_projection(x)  # (B,channels,K*L)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, diffusion_emb, cond_in)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x

    def prepare_output(self, synthesized):
        return synthesized.detach().cpu().numpy()
