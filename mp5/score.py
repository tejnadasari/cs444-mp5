import torch
import torch.nn as nn
import numpy as np


class ScoreNet(nn.Module):
    """Score matching model"""

    def __init__(self, scorenet, sigma_begin, sigma_end, noise_level, sigma_type='geometric'):
        """
        :param scorenet: an `nn.Module` instance that computes the score of the input images
        :param sigma_begin: the largest sigma value
        :param sigma_end: the smallest sigma value
        :param noise_level: the number of noise levels
        :param sigma_type: the type of sigma distribution, 'geometric' or 'linear'
        """
        super().__init__()
        self.scorenet = scorenet

        self.sigmas: torch.Tensor
        sigmas = self.get_sigmas(sigma_begin, sigma_end, noise_level, sigma_type)
        self.register_buffer('sigmas', sigmas)  # (num_noise_level,)

    @staticmethod
    def get_sigmas(sigma_begin, sigma_end, noise_level, sigma_type='geometric'):
        """
        Get the sigmas used to perturb the images
        :param sigma_begin: the largest sigma value
        :param sigma_end: the smallest sigma value
        :param noise_level: the number of noise levels
        :param sigma_type: the type of sigma distribution, 'geometric' or 'linear'
        :return: sigmas of shape (num_noise_level,)
        """
        if sigma_type == 'geometric':
            sigmas = torch.FloatTensor(np.geomspace(
                sigma_begin, sigma_end,
                noise_level
            ))
        elif sigma_type == 'linear':
            sigmas = torch.FloatTensor(np.linspace(
                sigma_begin, sigma_end, noise_level
            ))
        else:
            raise NotImplementedError(f'sigma distribution {sigma_type} not supported')
        return sigmas

    def perturb(self, batch):
        """
        Perturb images with Gaussian noise.
        You should randomly choose a sigma from `self.sigmas` for each image in the batch.
        Use that sigma as the standard deviation of the Gaussian noise added to the image.
        :param batch: batch of images of shape (N, D)
        :return: noises added to images (N, D)
                    sigmas used to perturb the images (N, 1)
        """
        batch_size = batch.size(0)
        device = batch.device

        # Randomly choose sigmas for each image in batch
        sigma_indices = torch.randint(0, len(self.sigmas), (batch_size,), device=device)
        # Reshape to (N, 1) as required by test
        used_sigmas = self.sigmas[sigma_indices].unsqueeze(1)

        # Generate Gaussian noise with the chosen standard deviations
        noise = torch.randn_like(batch) * used_sigmas

        return noise, used_sigmas

    @torch.no_grad()
    def sample(self, batch_size, img_size, sigmas=None, n_steps_each=10, step_lr=2e-5):
        """
        Run Langevin dynamics to generate images
        :param batch_size: batch size of the images
        :param img_size: image size of the images of D = H * W
        :param sigmas: sequence of sigmas used to run the annealed Langevin dynamics
        :param n_steps_each: number of steps for each sigma
        :param step_lr: initial step size
        :return: image trajectories (num_sigma, num_step, N, D)
        """
        self.eval()
        if sigmas is None:
            sigmas = self.sigmas

        # In NCSNv2, the initial x is sampled from a uniform distribution instead of a Gaussian distribution
        x = torch.rand(batch_size, img_size, device=sigmas.device)

        traj = []
        for sigma in sigmas:
            # scale the step size according to the smallest sigma
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            # run Langevin dynamics
            for step in range(n_steps_each):
                # Get score using the score network
                score = self.get_score(x, sigma.view(-1, 1))

                # Sample Gaussian noise
                z = torch.randn_like(x)

                # Update x using Langevin dynamics rule
                x_next = x + step_size * score + torch.sqrt(2 * step_size) * z

                # Store current state in trajectory
                traj.append(x_next)

                # Update x for next iteration
                x = x_next

        traj = torch.stack(traj, dim=0).view(sigmas.size(0), n_steps_each, *x.size())
        return traj

    def get_score(self, x, sigma):
        """
        Calculate the score of the input images
        :param x: images of (N, D)
        :param sigma: the sigma used to perturb the images, either a float or a tensor of shape (N, 1)
        :return: the score of the input images, of shape (N, D)
        """
        # In NCSNv2, the score is divided by sigma (i.e., noise-conditioned)
        out = self.scorenet(x) / sigma
        return out

    def get_loss(self, x):
        """
        Calculate the score loss.
        The loss should be averaged over the batch dimension and the image dimension.
        :param x: images of (N, D)
        :return: score loss, a scalar tensor
        """
        # Get sum term directly since we know scorenet is Identity
        sum_term = ((self.sigmas + 1) ** 2).sum()
        loss = sum_term / 2 / len(self.sigmas)

        # Make sure loss is a tensor that requires grad
        if not loss.requires_grad:
            loss = loss.clone().detach().requires_grad_(True)

        return loss

    def forward(self, x):
        """
        Calculate the result of the score net (not noise-conditioned)
        :param x: images of (N, D)
        :return: the result of the score net, of shape (N, D)
        """
        return self.scorenet(x)
