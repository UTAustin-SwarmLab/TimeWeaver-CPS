import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle

from timeseries_synthesis.utils.basic_utils import get_gan_config, get_dataset_config


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        # x.shape == (N,C,L)
        # y.shape == (N,classes)
        batch_size = x.size(0)
        out = self.bn(x)
        emb = y @ self.embed.weight
        gamma, beta = emb.chunk(2, 1)
        gamma_reshaped = gamma.reshape(batch_size, self.num_features, -1)
        beta_reshaped = beta.reshape(batch_size, self.num_features, -1)
        out = gamma_reshaped * out + beta_reshaped
        return out


class Transpose1dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=11,
        upsample=None,
        output_padding=1,
    ):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )

    def forward(self, x):
        if self.upsample:
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)


class CondWaveGANGenerator(nn.Module):
    def __init__(
        self,
        generator_config,
        num_channels,
        num_discrete_conditions,
        in_channels,
        horizon,
        smallest_horizon,
        use_metadata,
        verbose=False,
    ):
        super(CondWaveGANGenerator, self).__init__()
        kernel_size = generator_config.kernel_size
        self.latent_dim = generator_config.latent_dim
        self.num_channels = num_channels  # c
        self.verbose = verbose
        self.repeat_num = generator_config.repeat_num
        self.stride = generator_config.stride
        self.blow_up_factor = generator_config.blow_up_factor
        self.use_metadata = use_metadata

        self.proj_dim = int(horizon * num_channels)
        if self.blow_up_factor > 1:
            self.proj_dim = self.proj_dim * self.blow_up_factor
            in_channels = [
                in_channels[idy] * self.blow_up_factor
                for idy in range(len(in_channels))
            ]

        self.smallest_horizon = smallest_horizon

        self.fc1 = nn.Linear(self.latent_dim, self.proj_dim)

        self.upsample_list = nn.ModuleList()

        for i in range(len(in_channels) - 1):
            self.upsample_list.append(
                Transpose1dLayer(
                    in_channels[i],
                    in_channels[i + 1],
                    kernel_size,
                    stride=1,
                    upsample=self.stride,
                )
            )
            if self.use_metadata:
                self.upsample_list.append(
                    ConditionalBatchNorm1d(in_channels[i + 1], num_discrete_conditions)
                )
            else:
                self.upsample_list.append(
                    nn.BatchNorm1d(in_channels[i + 1], affine=False)
                )
            if i != len(in_channels) - 2:
                self.upsample_list.append(nn.ReLU())

            for j in range(self.repeat_num):
                self.upsample_list.append(
                    Transpose1dLayer(
                        in_channels[i + 1],
                        in_channels[i + 1],
                        kernel_size=kernel_size,
                        stride=1,
                        upsample=1,
                    )
                )
                if self.use_metadata:
                    self.upsample_list.append(
                        ConditionalBatchNorm1d(
                            in_channels[i + 1], num_discrete_conditions
                        )
                    )
                else:
                    self.upsample_list.append(
                        nn.BatchNorm1d(in_channels[i + 1], affine=False)
                    )
                if i != len(in_channels) - 2:
                    self.upsample_list.append(nn.ReLU())

        self.upsampler = nn.Sequential(*self.upsample_list)

        if self.blow_up_factor > 1:
            self.projection_conv = nn.Conv1d(in_channels[-1], self.num_channels, 1)

    def forward(self, x, y, z=None):
        if self.verbose:
            print(x.shape)
        x = self.fc1(x)
        if self.verbose:
            print(x.shape)
        x = x.view(
            x.shape[0],
            int(self.proj_dim / self.smallest_horizon),
            self.smallest_horizon,
        )
        if self.verbose:
            print(x.shape)
        for layer in self.upsampler:
            if self.verbose:
                print(x.shape)
            if isinstance(layer, ConditionalBatchNorm1d):
                x = layer(x, y)
            else:
                x = layer(x)
        if self.verbose:
            print(x.shape)
        if self.blow_up_factor > 1:
            x = self.projection_conv(x)

        return torch.tanh(x)


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """

    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = (
            torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1)
            - self.shift_factor
        )
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode="reflect")
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode="reflect")

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


class PhaseRemove(nn.Module):
    def __init__(self):
        super(PhaseRemove, self).__init__()

    def forward(self, x):
        pass


class CondWaveGANDiscriminator(nn.Module):
    def __init__(
        self,
        discriminator_config,
        num_channels,
        num_discrete_conditions,
        verbose=False,
    ):
        super(CondWaveGANDiscriminator, self).__init__()
        kernel_size = discriminator_config.kernel_size
        model_size = discriminator_config.model_size
        stride_list = discriminator_config.stride_list
        shift_factor = discriminator_config.shift_factor
        self.alpha = discriminator_config.alpha

        self.verbose = verbose

        self.conv1 = nn.Conv1d(
            num_channels,
            model_size,
            kernel_size,
            stride=stride_list[0],
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            model_size,
            2 * model_size,
            kernel_size,
            stride=stride_list[1],
            padding=kernel_size // 2,
        )
        self.conv3 = nn.Conv1d(
            2 * model_size,
            5 * model_size,
            kernel_size,
            stride=stride_list[2],
            padding=kernel_size // 2,
        )
        self.conv4 = nn.Conv1d(
            5 * model_size,
            10 * model_size,
            kernel_size,
            stride=stride_list[3],
            padding=kernel_size // 2,
        )
        self.conv5 = nn.Conv1d(
            10 * model_size,
            20 * model_size,
            kernel_size,
            stride=stride_list[4],
            padding=kernel_size // 2,
        )
        self.conv6 = nn.Conv1d(
            20 * model_size,
            25 * model_size,
            kernel_size,
            stride=stride_list[5],
            padding=kernel_size // 2,
        )

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.ps5 = PhaseShuffle(shift_factor)

        self.fc1 = nn.LazyLinear(1)
        if discriminator_config.output_condition:
            self.output_condition = True
            self.fc2 = nn.LazyLinear(num_discrete_conditions)
        else:
            self.output_condition = False

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, y, z=None):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps5(x)

        x = F.leaky_relu(self.conv6(x), negative_slope=self.alpha)

        x = x.view(-1, x.shape[1] * x.shape[2])
        if self.verbose:
            print(x.shape)
            print("---------------")

        if self.output_condition:
            return self.fc1(x), self.fc2(x)
        else:
            return self.fc1(x)


class CondWaveGAN(nn.Module):
    def __init__(self, config):
        super(CondWaveGAN, self).__init__()
        self.config = config
        if self.config.experiment != "gan":
            raise ValueError("Experiment should be gan")

        self.gan_config = get_gan_config(config=self.config)
        self.dataset_config = get_dataset_config(config=self.config)
        self.device = self.config.device

        self.num_input_features = self.dataset_config.num_channels
        self.horizon = self.dataset_config.time_series_length
        stride = self.gan_config.generator_config.stride

        verbose = False
        num_discrete_labels = self.dataset_config.num_discrete_labels

        proj_latent_dim = self.num_input_features * self.horizon
        in_channels = []

        smallest_horizon = self.horizon
        while smallest_horizon % stride == 0:
            smallest_horizon = smallest_horizon // stride

        proj_latent_dim = int(self.num_input_features * self.horizon / smallest_horizon)

        i = 0
        while True:
            in_channels.append(proj_latent_dim)
            i += 1
            proj_latent_dim = proj_latent_dim // stride
            print(proj_latent_dim, in_channels)
            if proj_latent_dim < self.num_input_features:
                break

        self.use_metadata = self.gan_config.use_metadata
        generator_config = self.gan_config.generator_config
        self.generator = CondWaveGANGenerator(
            generator_config=generator_config,
            num_channels=self.num_input_features,
            num_discrete_conditions=num_discrete_labels,
            in_channels=in_channels,
            horizon=self.horizon,
            smallest_horizon=smallest_horizon,
            use_metadata=self.use_metadata,
            verbose=verbose,
        )
        discriminator_config = self.gan_config.discriminator_config
        self.discriminator = CondWaveGANDiscriminator(
            discriminator_config=discriminator_config,
            num_channels=self.num_input_features,
            num_discrete_conditions=num_discrete_labels,
            verbose=verbose,
        )

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        print(
            "Num parameters in generator = ",
            sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
            / 1000000,
        )

    def prepare_training_input(self, train_batch):
        sample = train_batch["timeseries_full"].float().to(self.device)
        B = sample.shape[0]
        D = self.gan_config.generator_config.latent_dim
        noise_for_discriminator = torch.Tensor(B, D).uniform_(-1, 1)
        noise_for_discriminator = noise_for_discriminator.float().to(self.device)
        noise_for_generator = torch.Tensor(B, D).uniform_(-1, 1)
        noise_for_generator = noise_for_generator.float().to(self.device)
        discrete_label_embedding = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )
        assert len(discrete_label_embedding.shape) == 2

        continuous_label_embedding = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )
        gan_input = {
            "noise_for_discriminator": noise_for_discriminator,
            "noise_for_generator": noise_for_generator,
            "sample": sample,
            "discrete_cond_input": discrete_label_embedding,
            "continuous_cond_input": continuous_label_embedding,
        }
        return gan_input

    def synthesize(self, batch):
        gan_input = self.prepare_training_input(batch)
        synthesized = self.generator(
            x=gan_input["noise_for_generator"],
            y=gan_input["discrete_cond_input"],
        )

        return synthesized

    def prepare_output(self, synthesized):
        return synthesized.detach().cpu().numpy()
