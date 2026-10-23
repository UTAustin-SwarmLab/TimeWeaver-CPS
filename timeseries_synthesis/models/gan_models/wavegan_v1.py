import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle

from timeseries_synthesis.utils.basic_utils import get_gan_config, get_dataset_config

from timeseries_synthesis.models.diffusion_models.timeseries_diffusion_models.utils import (
    MetaDataEncoder,
)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class BatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, x):
        out = self.bn(x)
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
        dataset_config,
        in_channels,
        smallest_horizon,
        use_metadata,
        device,
        verbose=False,
    ):
        super(CondWaveGANGenerator, self).__init__()
        kernel_size = generator_config.kernel_size
        self.latent_dim = generator_config.latent_dim
        self.num_channels = dataset_config.num_channels  # c
        self.verbose = verbose
        self.repeat_num = generator_config.repeat_num
        self.stride = generator_config.stride
        self.blow_up_factor = generator_config.blow_up_factor
        self.use_metadata = use_metadata

        self.final_activation_str = generator_config.final_activation
        if self.final_activation_str == "tanh":
            self.final_activation = torch.nn.Tanh()
        elif self.final_activation_str == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
            print("Using sigmoid")
        elif self.final_activation_str == "None":
            self.final_activation = None

        self.proj_dim = int(dataset_config.time_series_length * self.num_channels)
        if self.blow_up_factor > 0:
            self.proj_dim = (
                self.proj_dim * self.blow_up_factor
            )  # a hyperparameter to match the model size to that of diffusion models
            in_channels = [
                in_channels[idy] * self.blow_up_factor
                for idy in range(len(in_channels))
            ]

        self.smallest_horizon = smallest_horizon

        self.fc1 = nn.Linear(self.latent_dim, self.proj_dim)
        if self.use_metadata:
            cond_in_dim = (
                generator_config.metadata_encoder_config.channels
                * dataset_config.time_series_length
            )
            self.fc2 = nn.Linear(cond_in_dim, self.proj_dim)

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
            self.upsample_list.append(BatchNorm1d(in_channels[i + 1]))
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

                self.upsample_list.append(BatchNorm1d(in_channels[i + 1]))
                if i != len(in_channels) - 2:
                    self.upsample_list.append(nn.ReLU())

        self.upsampler = nn.Sequential(*self.upsample_list)

        if self.blow_up_factor > 0:
            self.projection_conv = nn.Conv1d(in_channels[-1], self.num_channels, 1)

        if self.use_metadata:
            self.meta_data_encoder = MetaDataEncoder(
                dataset_config=dataset_config,
                denoiser_config=generator_config,
                device=device,
            )

    def forward(self, x, y, z=None):
        if self.verbose:
            print(x.shape)
        x = self.fc1(x)
        if self.verbose:
            print(x.shape)

        if self.use_metadata:
            cond_in = self.meta_data_encoder(y, z)
            cond_in = cond_in.reshape(cond_in.shape[0], -1)
            cond_in = self.fc2(cond_in)
        else:
            # print("here in gen")
            cond_in = torch.zeros_like(x).to(x.device)

        # print(x.shape, cond_in.shape)

        x = x + cond_in
        if self.verbose:
            print(x.shape)

        x = x.view(
            x.shape[0],
            int(self.proj_dim / self.smallest_horizon),
            self.smallest_horizon,
        )

        if self.verbose:
            print(x.shape)

        x = self.upsampler(x)
        if self.verbose:
            print(x.shape)
        if self.blow_up_factor > 0:
            x = self.projection_conv(x)

        if self.verbose:
            print(x.shape)

        if self.final_activation_str == "None":
            return x
        else:
            # print("final activation", self.final_activation_str)
            return self.final_activation(x)


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
        dataset_config,
        use_metadata,
        device,
        verbose=False,
    ):
        super(CondWaveGANDiscriminator, self).__init__()
        kernel_size = discriminator_config.kernel_size
        model_size = discriminator_config.model_size
        stride_list = discriminator_config.stride_list
        shift_factor = discriminator_config.shift_factor
        self.alpha = discriminator_config.alpha
        self.num_channels = dataset_config.num_channels
        self.use_metadata = use_metadata

        self.verbose = verbose

        if self.use_metadata:
            self.conv_combiner = nn.Conv1d(
                self.num_channels
                + discriminator_config.metadata_encoder_config.channels,
                5 * model_size,
                kernel_size,
                stride=stride_list[0],
                padding=kernel_size // 2,
            )
        else:
            self.conv_combiner = nn.Conv1d(
                self.num_channels,
                5 * model_size,
                kernel_size,
                stride=stride_list[0],
                padding=kernel_size // 2,
            )
        self.conv2 = nn.Conv1d(
            5 * model_size,
            5 * model_size,
            kernel_size,
            stride=stride_list[1],
            padding=kernel_size // 2,
        )
        self.conv3 = nn.Conv1d(
            5 * model_size,
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

        if self.use_metadata:
            self.metadata_encoder = MetaDataEncoder(
                dataset_config=dataset_config,
                denoiser_config=discriminator_config,
                device=device,
            )

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, y, z=None):
        if self.use_metadata:
            cond_in = self.metadata_encoder(y, z)
            cond_in = torch.einsum("ijk->ikj", cond_in)
            in_ = torch.cat([x, cond_in], dim=1)
        else:
            # print("here in disc")
            in_ = x

        x = F.leaky_relu(self.conv_combiner(in_), negative_slope=self.alpha)
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

        return self.fc1(x)


class CondWaveGAN_v1(nn.Module):
    def __init__(self, config):
        super(CondWaveGAN_v1, self).__init__()
        self.config = config
        self.gan_config = get_gan_config(config=self.config)
        self.dataset_config = get_dataset_config(config=self.config)
        self.device = self.config.device

        self.num_input_features = self.dataset_config.num_channels
        self.horizon = self.dataset_config.time_series_length
        stride = self.gan_config.generator_config.stride

        verbose = False

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

        generator_config = self.gan_config.generator_config

        self.use_metadata = self.gan_config.use_metadata
        self.generator = CondWaveGANGenerator(
            generator_config=generator_config,
            dataset_config=self.dataset_config,
            in_channels=in_channels,
            smallest_horizon=smallest_horizon,
            use_metadata=self.use_metadata,
            device=self.device,
            verbose=verbose,
        )
        discriminator_config = self.gan_config.discriminator_config
        self.discriminator = CondWaveGANDiscriminator(
            discriminator_config=discriminator_config,
            dataset_config=self.dataset_config,
            use_metadata=self.use_metadata,
            device=self.device,
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
        if len(discrete_label_embedding.shape) == 2:
            discrete_label_embedding = discrete_label_embedding.unsqueeze(1)
            discrete_label_embedding = discrete_label_embedding.repeat(
                1, sample.shape[2], 1
            )
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

        # if self.config.use_constraints:
        #     gan_input["equality_constraints"] = (
        #         train_batch["equality_constraints"].float().to(self.device)
        #     )

        return gan_input

    def synthesize(self, batch):
        gan_input = self.prepare_training_input(batch)
        synthesized = self.generator(
            x=gan_input["noise_for_generator"],
            y=gan_input["discrete_cond_input"],
            z=gan_input["continuous_cond_input"],
        )

        return synthesized

    def prepare_output(self, synthesized):
        return synthesized.detach().cpu().numpy()
