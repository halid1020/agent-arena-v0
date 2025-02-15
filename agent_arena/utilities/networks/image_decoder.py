from agent.utilities.torch_utils import *
from utilities.networks.residual_block import ResidualBlock

class ImageDecoder(nn.Module):
        __constants__ = ['embedding_size', 'image_dim']

        def __init__(self, embedding_size, image_dim, 
                     activation_function='relu', batchnorm=False, output_mode=None):
            super().__init__()
            self.image_dim = image_dim
            self.act_fn = getattr(F, activation_function)
            self.embedding_size = embedding_size
            #self.fc1 = nn.Linear(embedding_size)
            self.output_mode = output_mode
            if image_dim[1] == 128:
                self._decoder = nn.Sequential(
                    nn.ConvTranspose2d(embedding_size, 128, 5, stride=2),
                    nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(128, 64, 5, stride=2),
                    nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(64, 32, 5, stride=2),
                    nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(32, 16, 6, stride=2),
                    nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(16, self.image_dim[0], 6, stride=2)
                )
            elif image_dim[1] == 64:
                self._decoder = nn.Sequential(
                    nn.ConvTranspose2d(embedding_size, 128, 5, stride=2),
                    nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(128, 64, 5, stride=2),
                    nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(64, 32, 6, stride=2),
                    nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(32, self.image_dim[0], 6, stride=2)
                )


        def forward(self, z):
            batch_shape = z.shape[:-1]
            embed_size = z.shape[-1]
            squeezed_size = np.prod(batch_shape).item()
            z = z.reshape(squeezed_size, embed_size)

            # hidden = self.fc1(z)  # No nonlinearity here
            z = z.view(-1, self.embedding_size, 1, 1)
            #print('z shape', z.shape)
            x = self._decoder(z)

            shape = x.shape[1:]
            x = x.reshape((*batch_shape, *shape))

            if self.output_mode == 'normal':
                x = td.Independent(td.Normal(x, 1), len(shape))

            return x