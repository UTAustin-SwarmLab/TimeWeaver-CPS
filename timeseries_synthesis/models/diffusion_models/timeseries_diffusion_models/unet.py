import torch
import math
from torch import nn, einsum, Tensor
from functools import partial
import torch.nn.functional as F
from einops import rearrange, reduce
from timeseries_synthesis.utils.basic_utils import (
    get_denoiser_config,
    get_dataset_config,
)

from timeseries_synthesis.models.diffusion_models.timeseries_diffusion_models.utils import (
    MetaDataEncoder,
)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)


class Unet1D(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser_config = get_denoiser_config(config=self.config)
        self.dataset_config = get_dataset_config(config=self.config)
        self.device = self.config.device

        self.channels = self.denoiser_config.channels  # 64
        input_channels = self.dataset_config.num_channels
        kernel_size = self.denoiser_config.kernel_size  # 7
        padding = self.denoiser_config.padding  # 3
        dim_mults = tuple(self.denoiser_config.dim_mults)  # (1, 2, 4, 8)
        resnet_block_groups = self.denoiser_config.resnet_block_groups  # 8
        sinusoidal_pos_emb_theta = (
            self.denoiser_config.sinusoidal_pos_emb_theta
        )  # 10000
        attn_dim_head = self.denoiser_config.attn_dim_head  # 32
        attn_heads = self.denoiser_config.attn_heads  # 4

        self.metadata_encoder = MetaDataEncoder(
            dataset_config=self.dataset_config,
            denoiser_config=self.denoiser_config,
            device=self.device,
        )

        self.init_conv = torch.nn.Conv1d(
            self.channels + input_channels, self.channels, kernel_size, padding=padding
        )

        init_dim = self.channels
        dims = [init_dim, *map(lambda m: self.channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = self.channels * 4

        sinu_pos_emb = SinusoidalPosEmb(self.channels, theta=sinusoidal_pos_emb_theta)
        fourier_dim = self.channels

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(
            PreNorm(
                mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)
            )
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv1d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.out_dim = input_channels

        self.final_res_block = block_klass(
            self.channels * 2, self.channels, time_emb_dim=time_dim
        )
        self.final_conv = nn.Conv1d(self.channels, self.out_dim, 1)

        T = 200
        beta_0 = 0.0001
        beta_T = 0.1
        self.diffusion_hyperparameters = self.calc_diffusion_hyperparams(
            T=T,
            beta_0=beta_0,
            beta_T=beta_T,
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

    def prepare_training_input(self, train_batch):
        # sample
        sample = train_batch["timeseries_full"].float().to(self.device)
        assert sample.shape[1] == self.dataset_config.num_channels

        # discrete and continuous condition input
        discrete_label_embedding = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )
        if len(discrete_label_embedding.shape) == 2:
            discrete_label_embedding = discrete_label_embedding.unsqueeze(1)
            discrete_label_embedding = discrete_label_embedding.repeat(
                1, sample.shape[2], 1
            )
            assert (
                discrete_label_embedding[:, 0, :] == discrete_label_embedding[:, 1, :]
            ).all(), "Discrete label embedding is not being broadcasted correctly"
        assert (
            discrete_label_embedding.shape[1] == sample.shape[2]
        ), "Wrong shape for discrete_label_embedding"
        assert (
            discrete_label_embedding.shape[2] == self.dataset_config.num_discrete_labels
        ), "Wrong shape for discrete_label_embedding"

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
        ).to(
            self.device
        )  # random diffusion step

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

        return denoiser_input

    def forward(self, denoiser_input):
        x = denoiser_input["noisy_sample"]  # (B, K, L)
        # print(x.shape)
        cond_in = self.metadata_encoder(
            discrete_conditions=denoiser_input["discrete_cond_input"],
            continuous_conditions=denoiser_input["continuous_cond_input"],
        )
        cond_in = torch.einsum("b l c -> b c l", cond_in)
        x = torch.cat((x, cond_in), dim=1)
        x = self.init_conv(x)
        r = x.clone()

        diffusion_step = denoiser_input["diffusion_step"]
        diffusion_step = diffusion_step.long()
        # print(diffusion_step)
        t = self.time_mlp(diffusion_step)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        # print(x.shape)
        return self.final_conv(x)

    def prepare_output(self, synthesized):
        return synthesized.detach().cpu().numpy()
