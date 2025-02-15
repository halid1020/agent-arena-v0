from .gan_predictor import GAN_Predictor
from .unet_predictor import UNet_Predictor
from .pretrained_efficient_net_predictor \
    import PretrainedEfficientNetPredictor

PHASE_PREDICTOR = {
    'gan_predictor': GAN_Predictor,
    'unet_predictor': UNet_Predictor,
    'pretrained_efficient_net_predictor': PretrainedEfficientNetPredictor
}