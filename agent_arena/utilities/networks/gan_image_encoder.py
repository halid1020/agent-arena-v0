from agent.utilities.torch_utils import *
from utilities.networks.residual_block import ResidualBlock

class GANImageEncoder(nn.Module):
    __constants__ = ['embedding_size', 'image_dim']

    def __init__(self, embedding_size, image_dim, 
                 activation_function='relu', batchnorm=False, residual=False):
        super().__init__()
        self.image_dim = image_dim
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size

        if self.image_dim[1] == 128:
            self._encoder = nn.Sequential(
                nn.Conv2d(self.image_dim[0], 16, 4, stride=2) if not residual else ResidualBlock(self.image_dim[0], 16, kernel=4, stride=2),
                nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(16, 32, 4, stride=2)  if not residual else ResidualBlock(16, 32, kernel=4, stride=2),
                nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(32, 64, 4, stride=2)  if not residual else ResidualBlock(32, 64, kernel=4, stride=2),
                nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(64, 128, 4, stride=2) if not residual else ResidualBlock(64, 128, kernel=4, stride=2),
                nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(128, 256, 4, stride=2) if not residual else ResidualBlock(128, 256, kernel=4, stride=2),
                nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function]()
            )
        elif self.image_dim[1] == 64:
            self._encoder = nn.Sequential(
                nn.Conv2d(self.image_dim[0], 32, 4, stride=2) if not residual else ResidualBlock(self.image_dim[0], 32, kernel=4, stride=2),
                nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(32, 64, 4, stride=2) if not residual else ResidualBlock(32, 64, kernel=4, stride=2),
                nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(64, 128, 4, stride=2) if not residual else ResidualBlock(64, 128, kernel=4, stride=2),
                nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(128, 256, 4, stride=2) if not residual else ResidualBlock(128, 256, kernel=4, stride=2),
                nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function]()
            )
        else:
            raise NotImplementedError
        
        if embedding_size == 1024:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        #print('x shape', x.shape)
        batch_shape = x.shape[:-3]
        embed_size = x.shape[-3:]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape((squeezed_size, *embed_size))
        hidden = self._encoder(x)
        hidden = hidden.reshape(-1, 1024)
        output = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        shape = output.shape[1:]

        # print('batch_shape', batch_shape)
        # print('shape', shape)

        output = output.reshape((*batch_shape, *shape))

        return output