# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Optimized Actor-Critic for Running Task with Advanced Architecture Features.

This module implements an enhanced actor-critic network with:
- Orthogonal initialization for stable training
- LayerNorm for training stability
- PopArt adaptive value normalization for critic
- Fourier feature embeddings for periodic gait capture
- Residual connections for deeper network training

The network maintains the same input/output interface as the standard ActorCritic,
allowing seamless integration with existing training pipelines.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class PopArt(nn.Module):
    """PopArt adaptive value normalization.

    Adaptively normalizes the value function output to stabilize training
    when reward scales vary. Maintains running statistics of returns.

    Reference: "Adaptive Methods for Non-Stationary Reinforcement Learning"
    https://arxiv.org/abs/1809.04474
    """

    def __init__(self, input_dim: int, output_dim: int = 1, beta: float = 0.0003):
        super().__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Learnable linear layer
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        # Running statistics
        self.register_buffer("mu", torch.zeros(output_dim))
        self.register_buffer("sigma", torch.ones(output_dim))
        self.register_buffer("mu_new", torch.zeros(output_dim))
        self.register_buffer("sigma_new", torch.ones(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with orthogonal initialization."""
        nn.init.orthogonal_(self.weight, gain=1.0)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with denormalized output."""
        # Normalize output during training for stability
        normalized = torch.matmul(x, self.weight.t()) + self.bias
        # Denormalize using running statistics
        return normalized * self.sigma + self.mu

    def update_stats(self, targets: torch.Tensor):
        """Update running statistics based on batch targets."""
        with torch.no_grad():
            # Compute batch statistics
            mu_batch = targets.mean(dim=0)
            sigma_batch = targets.std(dim=0).clamp(min=1e-6)

            # Update running statistics with momentum
            self.mu_new = (1 - self.beta) * self.mu + self.beta * mu_batch
            self.sigma_new = (1 - self.beta) * self.sigma + self.beta * sigma_batch

            # Update weights and bias to preserve unnormalized outputs
            self.weight.data = self.weight.data * (self.sigma / self.sigma_new).view(-1, 1)
            self.bias.data = (self.bias.data * self.sigma + self.mu - self.mu_new) / self.sigma_new

            # Update running statistics
            self.mu.copy_(self.mu_new)
            self.sigma.copy_(self.sigma_new)

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets using current statistics."""
        return (targets - self.mu) / self.sigma.clamp(min=1e-6)


class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for capturing periodic patterns.

    Projects inputs into a higher dimensional space using random Fourier features,
    which helps the network learn periodic gait patterns more effectively.

    Reference: "Fourier Features Let Networks Learn High Frequency Functions"
    https://arxiv.org/abs/2006.10739
    """

    def __init__(self, input_dim: int, embed_dim: int = 64, num_freqs: int = 16, sigma: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs

        # Random Fourier frequencies (not learnable)
        self.register_buffer(
            "B",
            torch.randn(input_dim, num_freqs) * sigma
        )

        # Learnable projection to combine frequency features
        self.projection = nn.Linear(num_freqs * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize projection layer with orthogonal initialization."""
        nn.init.orthogonal_(self.projection.weight, gain=1.0)
        nn.init.constant_(self.projection.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature transformation."""
        # Project to Fourier space: [B, num_freqs]
        phase = 2 * math.pi * torch.matmul(x, self.B)
        # Concatenate sine and cosine: [B, num_freqs * 2]
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        # Project and normalize
        return self.layer_norm(self.projection(features))


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm for stable deep network training.

    Uses pre-activation design (norm -> activation -> linear) for better
    gradient flow.
    """

    def __init__(self, dim: int, activation: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.activation = activation
        self.layer_norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with orthogonal initialization."""
        nn.init.orthogonal_(self.linear1.weight, gain=math.sqrt(2))
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.orthogonal_(self.linear2.weight, gain=math.sqrt(2))
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual forward pass."""
        residual = x

        # Pre-activation
        out = self.layer_norm1(x)
        out = self.activation(out)
        out = self.linear1(out)
        out = self.dropout(out)

        out = self.layer_norm2(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.dropout(out)

        return out + residual


class RunningOptimizedActorCritic(nn.Module):
    """Optimized Actor-Critic network for running tasks.

    Architecture features:
    1. Fourier feature embedding for periodic gait pattern capture
    2. Deep residual network with LayerNorm for stable training
    3. PopArt adaptive value normalization for critic
    4. Orthogonal initialization throughout
    5. Optional phase conditioning for gait synchronization

    Maintains exact same interface as standard ActorCritic for seamless
    integration with existing training code.

    Args:
        num_actor_obs: Dimension of actor observations
        num_critic_obs: Dimension of critic observations
        num_actions: Number of action dimensions
        actor_hidden_dims: Hidden layer dimensions for actor
        critic_hidden_dims: Hidden layer dimensions for critic
        activation: Activation function type
        init_noise_std: Initial action noise standard deviation
        noise_std_type: Type of noise parameterization ('scalar' or 'log')
        use_fourier_features: Whether to use Fourier feature embedding
        fourier_embed_dim: Dimension of Fourier feature embedding
        fourier_num_freqs: Number of Fourier frequencies
        use_popart: Whether to use PopArt value normalization
        use_residual: Whether to use residual connections
        residual_dropout: Dropout rate for residual blocks
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        use_fourier_features: bool = True,
        fourier_embed_dim: int = 64,
        fourier_num_freqs: int = 16,
        use_popart: bool = True,
        use_residual: bool = True,
        residual_dropout: float = 0.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "RunningOptimizedActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        super().__init__()

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.use_fourier_features = use_fourier_features
        self.use_popart = use_popart
        self.use_residual = use_residual

        activation_module = resolve_nn_activation(activation)

        # Build actor network
        self.actor = self._build_actor(
            num_actor_obs,
            num_actions,
            actor_hidden_dims,
            activation_module,
            use_fourier_features,
            fourier_embed_dim,
            fourier_num_freqs,
            use_residual,
            residual_dropout,
        )

        # Build critic network
        self.critic = self._build_critic(
            num_critic_obs,
            critic_hidden_dims,
            activation_module,
            use_fourier_features,
            fourier_embed_dim,
            fourier_num_freqs,
            use_residual,
            residual_dropout,
        )

        # PopArt for value normalization
        if use_popart:
            self.popart = PopArt(critic_hidden_dims[-1] if critic_hidden_dims else num_critic_obs)
        else:
            self.popart = None

        print(f"RunningOptimizedActorCritic initialized:")
        print(f"  Actor: {self.actor}")
        print(f"  Critic: {self.critic}")
        print(f"  Use Fourier Features: {use_fourier_features}")
        print(f"  Use PopArt: {use_popart}")
        print(f"  Use Residual: {use_residual}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)

        # Initialize all parameters with orthogonal initialization
        self._orthogonal_init()

    def _build_actor(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: nn.Module,
        use_fourier: bool,
        fourier_embed_dim: int,
        fourier_num_freqs: int,
        use_residual: bool,
        dropout: float,
    ) -> nn.Module:
        """Build the actor network."""
        layers = []

        # Fourier feature embedding (optional)
        if use_fourier:
            self.actor_fourier = FourierFeatureEmbedding(
                input_dim, fourier_embed_dim, fourier_num_freqs
            )
            current_dim = fourier_embed_dim
        else:
            self.actor_fourier = None
            current_dim = input_dim

        # Input projection if using Fourier features
        if use_fourier and hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dims[0]))
            layers.append(nn.LayerNorm(hidden_dims[0]))
            layers.append(activation)
            current_dim = hidden_dims[0]

        # Hidden layers with residual connections or standard MLP
        for i in range(len(hidden_dims)):
            next_dim = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else output_dim

            if use_residual and i < len(hidden_dims) - 1:
                # Use residual block
                layers.append(ResidualBlock(hidden_dims[i], activation, dropout))
                # Projection if dimensions don't match
                if hidden_dims[i] != next_dim:
                    layers.append(nn.Linear(hidden_dims[i], next_dim))
                    if i < len(hidden_dims) - 2:  # Not the last hidden layer
                        layers.append(nn.LayerNorm(next_dim))
                        layers.append(activation)
                current_dim = next_dim
            else:
                # Standard layer
                if i == len(hidden_dims) - 1:
                    # Output layer - no activation
                    layers.append(nn.Linear(hidden_dims[i], output_dim))
                else:
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i]))
                    layers.append(nn.LayerNorm(hidden_dims[i]))
                    layers.append(activation)
                current_dim = hidden_dims[i]

        if not hidden_dims:
            # Direct mapping if no hidden layers
            layers.append(nn.Linear(input_dim, output_dim))

        return nn.Sequential(*layers)

    def _build_critic(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: nn.Module,
        use_fourier: bool,
        fourier_embed_dim: int,
        fourier_num_freqs: int,
        use_residual: bool,
        dropout: float,
    ) -> nn.Module:
        """Build the critic network."""
        layers = []

        # Fourier feature embedding (optional)
        if use_fourier:
            self.critic_fourier = FourierFeatureEmbedding(
                input_dim, fourier_embed_dim, fourier_num_freqs
            )
            current_dim = fourier_embed_dim
        else:
            self.critic_fourier = None
            current_dim = input_dim

        # Input projection if using Fourier features
        if use_fourier and hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dims[0]))
            layers.append(nn.LayerNorm(hidden_dims[0]))
            layers.append(activation)
            current_dim = hidden_dims[0]

        # Hidden layers with residual connections or standard MLP
        for i in range(len(hidden_dims)):
            next_dim = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else 1

            if use_residual and i < len(hidden_dims) - 1:
                # Use residual block
                layers.append(ResidualBlock(hidden_dims[i], activation, dropout))
                # Projection if dimensions don't match
                if hidden_dims[i] != next_dim:
                    layers.append(nn.Linear(hidden_dims[i], next_dim))
                    if i < len(hidden_dims) - 2:  # Not the last hidden layer
                        layers.append(nn.LayerNorm(next_dim))
                        layers.append(activation)
                current_dim = next_dim
            else:
                # Standard layer
                if i == len(hidden_dims) - 1:
                    # Output layer - no activation (PopArt handles normalization)
                    layers.append(nn.Linear(hidden_dims[i], next_dim))
                else:
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i]))
                    layers.append(nn.LayerNorm(hidden_dims[i]))
                    layers.append(activation)
                current_dim = hidden_dims[i]

        if not hidden_dims:
            # Direct mapping if no hidden layers
            layers.append(nn.Linear(input_dim, 1))

        return nn.Sequential(*layers)

    def _orthogonal_init(self):
        """Apply orthogonal initialization to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.shape[0] == self.num_actions or module.weight.shape[0] == 1:
                    # Output layers - smaller gain
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    # Hidden layers - standard gain
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def reset(self, dones=None):
        """Reset method for compatibility with recurrent policies."""
        pass

    def forward(self):
        """Forward method - not used directly."""
        raise NotImplementedError

    @property
    def action_mean(self):
        """Current action distribution mean."""
        return self.distribution.mean

    @property
    def action_std(self):
        """Current action distribution standard deviation."""
        return self.distribution.stddev

    @property
    def entropy(self):
        """Entropy of the action distribution."""
        return self.distribution.entropy().sum(dim=-1)

    def _process_actor_obs(self, observations: torch.Tensor) -> torch.Tensor:
        """Process actor observations with optional Fourier features."""
        if self.use_fourier_features and self.actor_fourier is not None:
            # Check if we need to skip Fourier for the first layer
            if isinstance(self.actor[0], nn.Linear):
                # First layer is linear, so we need to project Fourier features
                fourier_feat = self.actor_fourier(observations)
                # The actor network expects Fourier features as input
                return fourier_feat
        return observations

    def _process_critic_obs(self, observations: torch.Tensor) -> torch.Tensor:
        """Process critic observations with optional Fourier features."""
        if self.use_fourier_features and self.critic_fourier is not None:
            if isinstance(self.critic[0], nn.Linear):
                fourier_feat = self.critic_fourier(observations)
                return fourier_feat
        return observations

    def update_distribution(self, observations: torch.Tensor):
        """Update the action distribution given observations."""
        # Process observations
        processed_obs = self._process_actor_obs(observations)

        # Compute mean
        mean = self.actor(processed_obs)

        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        # Clamp std to avoid numerical issues
        std = torch.clamp(std, min=1e-6)

        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample an action from the policy.

        Args:
            observations: Observation tensor [batch_size, num_obs]

        Returns:
            Sampled actions [batch_size, num_actions]
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions.

        Args:
            actions: Action tensor [batch_size, num_actions]

        Returns:
            Log probabilities [batch_size]
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Get deterministic action for inference.

        Args:
            observations: Observation tensor [batch_size, num_obs]

        Returns:
            Mean actions [batch_size, num_actions]
        """
        processed_obs = self._process_actor_obs(observations)
        actions_mean = self.actor(processed_obs)
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate the value function.

        Args:
            critic_observations: Critic observation tensor [batch_size, num_critic_obs]

        Returns:
            Value estimates [batch_size, 1] or [batch_size]
        """
        processed_obs = self._process_critic_obs(critic_observations)
        value = self.critic(processed_obs)

        # Apply PopArt if enabled
        if self.use_popart and self.popart is not None:
            # PopArt expects [batch, features] format
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            value = self.popart(value)

        return value

    def update_popart_stats(self, returns: torch.Tensor):
        """Update PopArt statistics with new return values.

        This should be called during training to adapt the value normalization.

        Args:
            returns: Return values tensor [batch_size] or [batch_size, 1]
        """
        if self.use_popart and self.popart is not None:
            if returns.dim() == 1:
                returns = returns.unsqueeze(-1)
            self.popart.update_stats(returns)

    def normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns using PopArt statistics.

        Args:
            returns: Return values tensor

        Returns:
            Normalized returns
        """
        if self.use_popart and self.popart is not None:
            return self.popart.normalize_targets(returns)
        return returns

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with compatibility handling.

        Allows loading standard ActorCritic weights into optimized version
        by skipping incompatible keys.
        """
        # Filter out keys that don't exist in this model
        model_dict = self.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        if len(filtered_dict) != len(state_dict):
            missing = set(state_dict.keys()) - set(filtered_dict.keys())
            print(f"Warning: Skipping {len(missing)} incompatible keys: {missing}")

        super().load_state_dict(filtered_dict, strict=False)
        return True
