import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureCNNEncoder(nn.Module):
    """
    NatureCNN-style encoder used in Stable Baselines3 for image-based SAC.
    Input: (C, H, W), default (3, 84, 84)
    Output: feature vector of size `feature_dim` (default 512)
    """
    def __init__(self, obs_shape=(3, 84, 84), feature_dim=512):
        super().__init__()
        assert len(obs_shape) == 3, "Input must be 3D (C,H,W)"
        self.conv_net = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9x9 -> 7x7
            nn.ReLU()
        )

        # Compute flatten size
        with torch.no_grad():
            n_flatten = self.conv_net(torch.zeros(1, *obs_shape)).view(1, -1).size(1)

        self.fc = nn.Linear(n_flatten, feature_dim)

    def forward(self, obs):
        # Normalize image to [0,1]
        x = obs / 255.0
        x = self.conv_net(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return F.relu(x)

class NatureCNNEncoderRegressor(nn.Module):
    def __init__(
        self,
        obs_shape=(3, 84, 84),
        state_dim=45,
        feature_dim=512,
        conv_layers=None,
        use_decoder=False,
        decoder_layers=None,
    ):
        """
        Args:
            obs_shape: (C, H, W)
            state_dim: output state dimension
            feature_dim: intermediate feature dimension before regression
            conv_layers: list of dicts for encoder Conv2d blocks
            use_decoder: whether to build a decoder
            decoder_layers: list of dicts for transpose conv blocks, e.g.
                [
                    {"out_channels": 64, "kernel_size": 3, "stride": 1},
                    {"out_channels": 32, "kernel_size": 4, "stride": 2},
                    {"out_channels": 3,  "kernel_size": 8, "stride": 4}
                ]
        """
        super().__init__()
        assert len(obs_shape) == 3, "Input must be 3D (C,H,W)"
        C, H, W = obs_shape

        # default NatureCNN encoder
        if conv_layers is None:
            conv_layers = [
                {"out_channels": 32, "kernel_size": 8, "stride": 4},
                {"out_channels": 64, "kernel_size": 4, "stride": 2},
                {"out_channels": 64, "kernel_size": 3, "stride": 1},
            ]

        # ----- ENCODER -----
        conv_modules = []
        in_channels = C
        for layer_cfg in conv_layers:
            conv_modules += [
                nn.Conv2d(
                    in_channels,
                    layer_cfg["out_channels"],
                    kernel_size=layer_cfg["kernel_size"],
                    stride=layer_cfg["stride"],
                ),
                nn.ReLU(),
            ]
            in_channels = layer_cfg["out_channels"]

        self.conv_net = nn.Sequential(*conv_modules)

        with torch.no_grad():
            n_flatten = self.conv_net(torch.zeros(1, *obs_shape)).view(1, -1).size(1)

        self.fc = nn.Linear(n_flatten, feature_dim)
        self.regressor = nn.Linear(feature_dim, state_dim)

        # ----- DECODER -----
        self.use_decoder = use_decoder
        if use_decoder:
            # If no decoder provided, mirror the NatureCNN in reverse
            if decoder_layers is None:
                decoder_layers = [
                    {"out_channels": 64, "kernel_size": 3, "stride": 1},
                    {"out_channels": 32, "kernel_size": 4, "stride": 2},
                    {"out_channels": C,  "kernel_size": 8, "stride": 4},
                ]

            # Determine pre-decoder feature map shape (reverse of conv)
            self.decoder_input_shape = self.conv_net(
                torch.zeros(1, *obs_shape)
            ).shape[1:]

            # Linear layer to expand embedding back to feature map
            decoder_input_dim = (
                self.decoder_input_shape[0]
                * self.decoder_input_shape[1]
                * self.decoder_input_shape[2]
            )
            self.fc_decoder = nn.Linear(feature_dim, decoder_input_dim)

            # Build transpose conv decoder
            deconv_modules = []
            in_channels = self.decoder_input_shape[0]
            for layer_cfg in decoder_layers:
                deconv_modules += [
                    nn.ConvTranspose2d(
                        in_channels,
                        layer_cfg["out_channels"],
                        kernel_size=layer_cfg["kernel_size"],
                        stride=layer_cfg["stride"],
                    ),
                    nn.ReLU() if layer_cfg["out_channels"] != C else nn.Sigmoid(),
                ]
                in_channels = layer_cfg["out_channels"]

            self.decoder = nn.Sequential(*deconv_modules)

    def forward(self, obs): ## assume rgb images
        # ---- Encoder ----
        x = obs / 255.0
        x = self.conv_net(x)
        x = x.reshape(x.size(0), -1)

        e = F.relu(self.fc(x))        # embedding
        state_pred = self.regressor(e)

        # ---- Optional Decoder ----
        if self.use_decoder:
            d = self.fc_decoder(e)
            d = d.view(
                x.size(0),
                self.decoder_input_shape[0],
                self.decoder_input_shape[1],
                self.decoder_input_shape[2],
            )
            recon = self.decoder(d)   # reconstructed image 0–1
            return e, state_pred, recon

        return e, state_pred, None
    
# class NatureCNNEncoderRegressor(nn.Module):
#     def __init__(self, obs_shape=(3, 84, 84), state_dim=45, feature_dim=512):
#         super().__init__()
#         assert len(obs_shape) == 3, "Input must be 3D (C,H,W)"
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20x20 -> 9x9
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9x9 -> 7x7
#             nn.ReLU()
#         )

#         # Compute flatten size
#         with torch.no_grad():
#             n_flatten = self.conv_net(torch.zeros(1, *obs_shape)).view(1, -1).size(1)

#         self.fc = nn.Linear(n_flatten, feature_dim)

#         self.regressor = nn.Linear(feature_dim, state_dim)

#     def forward(self, obs):
#         # Normalize image to [0,1]
#         x = obs / 255.0
#         x = self.conv_net(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc(x)
#         e = F.relu(x)
#         x = self.regressor(e)
#         return e, x


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, cnn_channels, out_dim, kernel=3, pool=2):
        super().__init__()
        layers = []
        prev = in_channels
        for c in cnn_channels:
            layers.append(nn.Conv2d(prev, c, kernel_size=kernel, stride=1, padding=kernel // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pool))
            prev = c
        self.cnn = nn.Sequential(*layers)
        self._project = None
        self._initialized = False
        self.out_dim = out_dim

    def _ensure_init(self, x: torch.Tensor):
        if self._initialized:
            return
        with torch.no_grad():
            y = self.cnn(x)
            sz = y.shape[1] * y.shape[2] * y.shape[3]
            self._project = nn.Sequential(nn.Flatten(), nn.Linear(sz, self.out_dim), nn.ReLU()).to(x.device)
            self._initialized = True

    def forward(self, contexts):
        # if contexts.dim() == 4:
        #     contexts = contexts.unsqueeze(0)
        B, N, C, H, W = contexts.shape
        contexts_flat = contexts.reshape(B, N * C, H, W)
        #print(contexts_flat.shape)
        self._ensure_init(contexts_flat)
        features = self.cnn(contexts_flat)
        features = features.view(features.size(0), -1)
        projected = self._project(features)
        projected = projected.view(B, -1)
        return projected #.mean(dim=1)


class MLPActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x)