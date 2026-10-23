import torch
from timeseries_synthesis.utils.basic_utils import (
    OKBLUE,
    ENDC,
)


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels, activation="gelu"
    )
    return torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)


class DiscreteFCEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, initial_projection_dim, projection_dim, dropout):
        super(DiscreteFCEncoder, self).__init__()
        self.projection1 = torch.nn.Linear(embedding_dim, initial_projection_dim)
        self.projection2 = torch.nn.Linear(initial_projection_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection1(inp)
        projected = self.gelu(projected)
        projected = self.projection2(projected)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class FCEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super(FCEncoder, self).__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection(inp)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ProjectionHead(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super(ProjectionHead, self).__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, inp):
        projected = self.projection(inp)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class MetaDataEncoder(torch.nn.Module):
    def __init__(self, dataset_config, denoiser_config, device):
        super(MetaDataEncoder, self).__init__()
        self.device = device
        self.dataset_config = dataset_config
        self.denoiser_config = denoiser_config

        num_discrete_conditions = self.dataset_config.num_discrete_conditions
        num_discrete_labels = self.dataset_config.num_discrete_labels
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

        self.sa_layer_exists = (
            True if self.denoiser_config.metadata_encoder_config.use_sa_layer else False
        )

        if self.discrete_condition_encoder_exists:
            self.discrete_condition_encoder = DiscreteFCEncoder(
                embedding_dim=num_discrete_labels,
                initial_projection_dim=int(
                    num_discrete_conditions
                    * self.dataset_config.discrete_condition_embedding_dim
                ),
                projection_dim=self.denoiser_config.metadata_encoder_config.channels,
                dropout=self.denoiser_config.metadata_encoder_config.dropout,
            )
            print(OKBLUE + "Discrete condition encoder exists" + ENDC)
            print(
                OKBLUE
                + "Discrete condition encoder input size = %d" % num_discrete_labels
                + ENDC
            )

        if self.continuous_condition_encoder_exists:
            self.continuous_condition_encoder = FCEncoder(
                embedding_dim=num_continuous_labels,
                projection_dim=self.denoiser_config.metadata_encoder_config.channels,
                dropout=0.1,
            )
            print(OKBLUE + "Continuous condition encoder exists" + ENDC)
            print(
                OKBLUE
                + "Continuous condition encoder input size = %d" % num_continuous_labels
                + ENDC
            )

        if self.combined_condition_encoder_exists:
            self.combined_condition_encoder = FCEncoder(
                embedding_dim=self.denoiser_config.metadata_encoder_config.channels * 2,
                projection_dim=self.denoiser_config.metadata_encoder_config.channels,
                dropout=0.1,
            )
            print(OKBLUE + "Combined condition encoder exists" + ENDC)

        if self.sa_layer_exists:
            self.condition_transformer_encoder = get_torch_trans(
                heads=self.denoiser_config.metadata_encoder_config.n_heads,
                layers=self.denoiser_config.metadata_encoder_config.num_encoder_layers,
                channels=self.denoiser_config.metadata_encoder_config.channels,
            )

    def position_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, discrete_conditions, continuous_conditions):
        B = discrete_conditions.shape[0]  # B
        L = self.dataset_config.time_series_length  # L

        # assert (discrete_conditions[:,0,:] == discrete_conditions[:,1,:]).all(), "Discrete conditions are not the same for all time steps"

        if self.discrete_condition_encoder_exists:
            discrete_conditions = self.discrete_condition_encoder(discrete_conditions)
            # assert (discrete_conditions[:,0,:] == discrete_conditions[:,1,:]).all(), "Discrete conditions are not the same for all time steps"
        if self.continuous_condition_encoder_exists:
            continuous_conditions = self.continuous_condition_encoder(
                continuous_conditions
            )
            if len(continuous_conditions.shape) == 2:
                continuous_conditions = continuous_conditions.unsqueeze(1).repeat(1, L, 1)
        if self.combined_condition_encoder_exists:
            combined_conditions = torch.cat(
                [discrete_conditions, continuous_conditions], dim=-1
            )
            combined_conditions = self.combined_condition_encoder(combined_conditions)
        else:
            combined_conditions = (
                discrete_conditions
                if self.discrete_condition_encoder_exists
                else continuous_conditions
            )

        if self.sa_layer_exists:
            tp = torch.arange(L).unsqueeze(0).repeat(B, 1).float().to(self.device)
            pos_emb = self.position_embedding(
                tp, self.denoiser_config.metadata_encoder_config.channels
            )
            combined_conditions = combined_conditions + pos_emb
            metadata_enc = self.condition_transformer_encoder(
                combined_conditions.permute(1, 0, 2)
            )  # L, B, C, for the transformer to act across the time dimension
            metadata_enc = metadata_enc.permute(1, 0, 2)
            return metadata_enc  # B, L, C
        else:
            return combined_conditions  # B, L, C
