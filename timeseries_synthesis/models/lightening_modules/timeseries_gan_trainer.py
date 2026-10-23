from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import os
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple
import numpy as np
import lightning as L

from timeseries_synthesis.models.load_models import load_timeseries_gan
from timeseries_synthesis.utils.basic_utils import get_dataset_config, get_gan_config


class TimeSeriesGANModelTrainer(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.automatic_optimization = False  # must for GANs
        self.config = config

        self.dataset_config = get_dataset_config(config=config)
        self.gan_config = get_gan_config(config=config)

        self.synthesizer = load_timeseries_gan(config=config)
        self.train_G_flag = True

        self.classification_loss_criterion = (
            torch.nn.BCEWithLogitsLoss()
        )  # because we have a multi-class classification problem

    def configure_optimizers(self):
        lr = self.config.training.learning_rate
        b1 = self.config.training.b1
        b2 = self.config.training.b2

        opt_g = torch.optim.Adam(
            self.synthesizer.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        opt_d = torch.optim.Adam(
            self.synthesizer.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        return [opt_g, opt_d], []

    def calc_gradient_penalty(
        self,
        real_data,
        fake_data,
        discrete_cond_input,
        continuous_cond_input,
        batch_size,
    ):
        # Compute interpolation factors
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.config.device)

        # Interpolate between real and fake data.
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.to(self.config.device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        # Evaluate discriminator
        if self.gan_config.discriminator_config.output_condition:
            # only for p2p and wavegan
            disc_interpolates, _ = self.synthesizer.discriminator(
                x=interpolates, y=discrete_cond_input, z=continuous_cond_input
            )
        else:
            disc_interpolates = self.synthesizer.discriminator(
                x=interpolates, y=discrete_cond_input, z=continuous_cond_input
            )

        # Obtain gradients of the discriminator with respect to the inputs
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(self.config.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
        # to be a 1-Lipschitz function.
        gradient_penalty = (
            self.config.training.lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        )
        return gradient_penalty

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        gan_input = self.synthesizer.prepare_training_input(batch)
        optimizer_g, optimizer_d = self.optimizers()

        if (batch_idx + 1) % self.config.training.train_generator_every == 0:
            self.train_G_flag = True

        for p in self.synthesizer.discriminator.parameters():
            p.requires_grad = True

        one = torch.tensor(1, dtype=torch.float)
        neg_one = one * -1

        one = one.to(self.config.device)
        neg_one = neg_one.to(self.config.device)

        #############################
        # (1) Train Discriminator
        #############################

        real_sample = gan_input["sample"].to(self.config.device)
        batch_size = real_sample.size(0)
        self.synthesizer.discriminator.zero_grad()

        # Noise
        noise = gan_input["noise_for_discriminator"].to(self.config.device)
        noise_Var = torch.autograd.Variable(noise, requires_grad=False)

        # a) compute loss contribution from real training data
        if self.gan_config.discriminator_config.output_condition:
            # if we are also predicting the class - mainly for electricity and ecg datasets, then we compute the classification loss for real data
            D_real, D_real_class = self.synthesizer.discriminator(
                x=real_sample,
                y=gan_input["discrete_cond_input"],
                z=gan_input["continuous_cond_input"],
            )
            real_classification_loss = self.classification_loss_criterion(
                D_real_class, gan_input["discrete_cond_input"]
            )
            self.manual_backward(
                real_classification_loss, retain_graph=True
            )  # real data classification loss backward

        else:
            D_real = self.synthesizer.discriminator(
                x=real_sample,
                y=gan_input["discrete_cond_input"],
                z=gan_input["continuous_cond_input"],
            )

        D_real = D_real.mean()  # avg loss
        D_real.backward(
            neg_one, retain_graph=True
        )  # loss * -1, real data real/fake classification backward

        # b) compute loss contribution from generated data, then backprop.
        fake = torch.autograd.Variable(
            self.synthesizer.generator(
                x=noise_Var,
                y=gan_input["discrete_cond_input"],
                z=gan_input["continuous_cond_input"],
            ).data
        )

        if self.gan_config.discriminator_config.output_condition:
            if self.gan_config.discriminator_config.predict_condition_for_fake:
                # if we are also predicting the class - mainly for electricity and ecg datasets, then we compute the classification loss for fake data
                D_fake, D_fake_class = self.synthesizer.discriminator(
                    x=fake,
                    y=gan_input["discrete_cond_input"],
                    z=gan_input["continuous_cond_input"],
                )
                fake_classification_loss = self.classification_loss_criterion(
                    D_fake_class, gan_input["discrete_cond_input"]
                )
                self.manual_backward(
                    fake_classification_loss, retain_graph=True
                )  # fake data classification loss backward
            else: 
                D_fake, _ = self.synthesizer.discriminator(
                    x=fake,
                    y=gan_input["discrete_cond_input"],
                    z=gan_input["continuous_cond_input"],
                )

        else:
            D_fake = self.synthesizer.discriminator(
                x=fake,
                y=gan_input["discrete_cond_input"],
                z=gan_input["continuous_cond_input"],
            )

        D_fake = D_fake.mean()
        D_fake.backward(
            one, retain_graph=True
        )  # loss * 1, fake data real/fake classification backward

        # c) compute gradient penalty and backprop
        gradient_penalty = self.calc_gradient_penalty(
            real_sample,
            fake.data,
            gan_input["discrete_cond_input"],
            gan_input["continuous_cond_input"],
            batch_size,
        )
        gradient_penalty.backward(one, retain_graph=True)  # gradient penalty backward

        # Compute cost * Wassertein loss..
        D_cost_train = D_fake - D_real + gradient_penalty
        D_wass_train = D_real - D_fake

        # Update gradient of discriminator.
        optimizer_d.step()  # update discriminator parameters

        D_cost_train_cpu = D_cost_train.data.cpu()
        D_wass_train_cpu = D_wass_train.data.cpu()

        # fake_classification_loss = fake_classification_loss.data.cpu()

        self.log(
            "d_loss",
            D_cost_train_cpu,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "w_dist",
            D_wass_train_cpu,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.gan_config.discriminator_config.output_condition:
            real_classification_loss = real_classification_loss.data.cpu()
            self.log(
                "real_classification_loss",
                real_classification_loss,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        #############################
        # (3) Train Generator
        #############################
        # print(self.train_G_flag)
        if self.train_G_flag:
            # Prevent discriminator update.
            for p in self.synthesizer.discriminator.parameters():
                p.requires_grad = False

            # Reset generator gradients
            self.synthesizer.generator.zero_grad()

            # Noise
            noise = gan_input["noise_for_generator"].to(self.config.device)
            noise = noise.to(self.config.device)
            noise_Var = torch.autograd.Variable(noise, requires_grad=False)

            fake = self.synthesizer.generator(
                x=noise_Var,
                y=gan_input["discrete_cond_input"],
                z=gan_input["continuous_cond_input"],
            )
            if self.gan_config.discriminator_config.output_condition:
                # if we are also predicting the class - mainly for electricity and ecg datasets, then we compute the classification loss for the generated data
                G, G_class = self.synthesizer.discriminator(
                    x=fake,
                    y=gan_input["discrete_cond_input"],
                    z=gan_input["continuous_cond_input"],
                )
                G = G.mean()
                gen_fake_classification_loss = self.classification_loss_criterion(
                    G_class, gan_input["discrete_cond_input"]
                )
                self.manual_backward(
                    gen_fake_classification_loss, retain_graph=True
                )  # generated data classification loss backward
            else:
                G = self.synthesizer.discriminator(
                    x=fake,
                    y=gan_input["discrete_cond_input"],
                    z=gan_input["continuous_cond_input"],
                )
                G = G.mean()

            # Update gradients.
            G.backward(
                neg_one, retain_graph=True
            )  # want to maximize the value of G, so we use -1 * loss
            G_cost = -G

            optimizer_g.step()  # update generator parameters
            G_cost_cpu = G_cost.data.cpu()

            self.log(
                "g_loss",
                G_cost_cpu,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            if self.gan_config.discriminator_config.output_condition:
                gen_fake_classification_loss = gen_fake_classification_loss.data.cpu()
                self.log(
                    "gen_fake_classification_loss",
                    gen_fake_classification_loss,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

            self.train_G_flag = False

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:

        gan_input = self.synthesizer.prepare_training_input(batch)
        for p in self.synthesizer.discriminator.parameters():
            p.requires_grad = False

        # Noise
        noise = gan_input["noise_for_generator"].to(self.config.device)
        noise = noise.to(self.config.device)
        noise_Var = torch.autograd.Variable(noise, requires_grad=False)

        fake = self.synthesizer.generator(
            x=noise_Var,
            y=gan_input["discrete_cond_input"],
            z=gan_input["continuous_cond_input"],
        )
        if self.gan_config.discriminator_config.output_condition:
            G, G_class = self.synthesizer.discriminator(
                x=fake,
                y=gan_input["discrete_cond_input"],
                z=gan_input["continuous_cond_input"],
            )
            gen_fake_classification_loss = self.classification_loss_criterion(
                G_class, gan_input["discrete_cond_input"]
            )
            gen_fake_classification_loss = gen_fake_classification_loss.data.cpu()
            self.log(
                "gen_fake_classification_loss",
                gen_fake_classification_loss,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            G = self.synthesizer.discriminator(
                x=fake,
                y=gan_input["discrete_cond_input"],
                z=gan_input["continuous_cond_input"],
            )
        G = G.mean()

        G_cost = -G

        G_cost_cpu = G_cost.data.cpu()

        self.log(
            "g_loss",
            G_cost_cpu,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
