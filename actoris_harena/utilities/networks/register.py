from utilities.networks.gan_image_encoder import GANImageEncoder
from utilities.networks.resnet import ResNet36_4s, ResNet43_8s, ResNet67_12s
from utilities.networks.unet import UNet

Name2Network = {
    'gan-encoder': GANImageEncoder,
    'resnet36': ResNet36_4s,
    'resnet43': ResNet43_8s,
    'resnet67': ResNet67_12s,
    'unet': UNet
}