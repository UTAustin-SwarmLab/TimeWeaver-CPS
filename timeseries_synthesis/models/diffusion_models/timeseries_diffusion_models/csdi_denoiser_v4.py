import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from time import sleep
import tqdm

from timeseries_synthesis.utils.basic_utils import (
    get_denoiser_config,
    get_dataset_config,
)

from timeseries_synthesis.models.diffusion_models.timeseries_diffusion_models.utils import (
    MetaDataEncoder,
)


def cosine_schedule(num_timesteps, s=0.008):
    def f(t):
        return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2

    x = torch.linspace(0, num_timesteps, num_timesteps + 1)
    alphas_cumprod = f(x) / f(torch.tensor([0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clip(betas, 0.0001, 0.999)
    return betas


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ConvLayer(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super(ConvLayer, self).__init__()
        self.downConv = torch.nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = torch.nn.BatchNorm1d(c_out)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, c_in, horizon).
        Returns:
            Output tensor of shape (batch_size, c_out, horizon).
        """
        x = self.downConv(x)  # (batch_size, d_model, seq_len)
        x = self.norm(x)
        x = self.activation(x)
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
        cond_in = cond_in.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
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
        self.embedding_dim = embedding_dim
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.diffusion_embedding[diffusion_step]
        # diffusion_step_index = (
        #     diffusion_step.unsqueeze(-1)
        #     .repeat(1, self.embedding_dim)
        #     .to(diffusion_step.device)
        # )
        # x = torch.gather(self.diffusion_embedding, 0, diffusion_step_index)
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


class CSDITSDenoiser_v4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser_config = get_denoiser_config(config=self.config)
        self.dataset_config = get_dataset_config(config=self.config)
        self.device = self.config.device

        # for each timestep in the timeseries, we have a positional embedding
        self.pos_embedding_dim = self.denoiser_config.positional_embedding_dim  # 128
        self.channels = self.denoiser_config.channels  # 512

        self.num_input_channels = self.dataset_config.num_channels  # K
        self.channel_embedding = torch.nn.Embedding(
            num_embeddings=self.num_input_channels,
            embedding_dim=self.denoiser_config.channel_embedding_dim,
        )  # 16

        # metadata encoder
        if self.denoiser_config.use_metadata:
            self.metadata_encoder = MetaDataEncoder(
                dataset_config=self.dataset_config,
                denoiser_config=self.denoiser_config,
                device=self.device,
            )

        self.input_projection = ConvLayer(
            c_in=1, c_out=self.channels
        )  # **** always c_in=1 ****

        self.output_projection = torch.nn.Sequential(
            Conv1d_with_init(self.channels, self.channels, 1),
            torch.nn.ReLU(),
            Conv1d_with_init(self.channels, 1, 1),
        )

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

        T = 200
        beta_0 = 0.0001
        beta_T = 0.1
        schedule = self.config.training.schedule
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=T,
            embedding_dim=self.channels,
        )
        self.diffusion_hyperparameters = self.calc_diffusion_hyperparams(
            T=T,
            beta_0=beta_0,
            beta_T=beta_T,
            schedule=schedule,
        )

    def calc_diffusion_hyperparams(self, T, beta_0, beta_T, schedule):
        if schedule == "linear":
            Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
            Alpha = 1 - Beta
            Alpha_bar = Alpha + 0
            Beta_tilde = Beta + 0
            for t in range(1, T): 
                Alpha_bar[t] *= Alpha_bar[t - 1]
                Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
            Sigma = torch.sqrt(Beta_tilde)
        elif schedule == "cosine":
            Beta = cosine_schedule(T)
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
            1, 1, self.num_input_channels, 1
        )  # (B, L, K, emb)
        feature_embed = self.channel_embedding(
            torch.arange(self.num_input_channels).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, emb, K, L  )
        return side_info.to(self.device)

    def prepare_training_input(self, train_batch):
        # sample
        sample = train_batch["timeseries_full"].float().to(self.device)
        assert sample.shape[1] == self.num_input_channels

        # discrete and continuous condition input
        discrete_label_embedding = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )
        # broadcast discrete label embedding if needed
        if len(discrete_label_embedding.shape) == 2:
            discrete_label_embedding = discrete_label_embedding.unsqueeze(1)
            discrete_label_embedding = discrete_label_embedding.repeat(
                1, sample.shape[2], 1
            )
            assert (
                discrete_label_embedding[:, 0, :] == discrete_label_embedding[:, 1, :]
            ).all(), "Discrete label embedding is not being broadcasted correctly"
        continuous_label_embedding = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )

        # diffusion step
        _dh = self.diffusion_hyperparameters
        B = sample.shape[0]
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
        t = torch.randint(
            0,
            T,
            (B,),
        )

        # noise and noisy data

        current_alpha_bar = Alpha_bar[t].unsqueeze(1).unsqueeze(1).to(self.device)
        noise = torch.randn_like(sample).float().to(self.device)
        noisy_data = (
            torch.sqrt(current_alpha_bar) * sample
            + torch.sqrt(1.0 - current_alpha_bar) * noise
        )
        denoiser_input = {
            "sample": sample,
            "noisy_sample": noisy_data,
            "noise": noise,
            "discrete_cond_input": discrete_label_embedding,
            "continuous_cond_input": continuous_label_embedding,
            "diffusion_step": t,
        }

        # if self.config.use_constraints:
        #     denoiser_input["equality_constraints"] = (
        #         train_batch["equality_constraints"].float().to(self.device)
        #     )

        return denoiser_input

    def forward(self, denoiser_input):
        noisy_input = denoiser_input["noisy_sample"]  # (B, K, L)

        B = noisy_input.shape[0]  # B
        K = noisy_input.shape[1]  # K
        L = noisy_input.shape[2]  # L

        tp = torch.arange(L).unsqueeze(0).repeat(B, 1).float().to(self.device)
        # assert (
        #     tp[0, :] == torch.arange(L).float().to(self.device)
        # ).all(), "Time points are not being broadcasted correctly"
        # assert (
        #     tp[0, :] == tp[1, :]
        # ).all(), "Time points are not being broadcasted correctly"
        side_info = self.get_side_info(tp)

        if self.denoiser_config.use_metadata:
            cond_in = self.metadata_encoder(
                discrete_conditions=denoiser_input["discrete_cond_input"],
                continuous_conditions=denoiser_input["continuous_cond_input"],
            )
        else:
            cond_in = torch.zeros(B, L, self.channels).to(self.device)
        # dummy = cond_in[0, 0, :]
        cond_in = torch.einsum("blc->bcl", cond_in)  # (B,channels,L)
        # assert (
        #     dummy == cond_in[0, :, 0]
        # ).all(), "Condition input is not being broadcasted correctly"

        cond_in = cond_in.unsqueeze(2).repeat(1, 1, K, 1)  # (B,channels,K,L)
        # assert (
        #     cond_in[:, :, 0, :] == cond_in[:, :, 1, :]
        # ).all(), "Condition input is not being broadcasted correctly"

        x = noisy_input.reshape(B * K, L).unsqueeze(1)  # (B*K,1,L)
        # assert (
        #     x[:K, 0] == noisy_input[0]
        # ).all(), "Noisy input is not being broadcasted correctly"
        x = self.input_projection(x)  # (B*K,channels,L)
        # dummy = x
        x = x.reshape(B, K, self.channels, L)
        # assert (
        #     x[0] == dummy[:K]
        # ).all(), "Input projection is not being broadcasted correctly"
        # dummy = x
        x = torch.einsum("bkcl->bckl", x)  # (B,channels,K,L)
        # assert (
        #     dummy[:, :, 0, :] == x[:, 0, :, :]
        # ).all(), "Input projection is not being broadcasted correctly"
        # assert (
        #     dummy[:, 0, :, :] == x[:, :, 0, :]
        # ).all(), "Input projection is not being broadcasted correctly"

        diffusion_step = denoiser_input["diffusion_step"].long().to(self.device)  # (B,)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, diffusion_emb, cond_in)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection(x)  # (B,channel,K*L)
        x = x.reshape(B, K, L)
        return x

    def prepare_output(self, synthesized):
        return synthesized.detach().cpu().numpy()
