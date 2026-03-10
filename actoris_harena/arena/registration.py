from .raven.raven_env_adapter import RavenEnvAdapter
from .raven.raven_pixel_env_adapter import RavenPixelEnvAdapter
from .raven.raven_pixel_env_adapter import RavenPixelEnvAdapterRay
from .gymnasium.wrapper import GymnasiumArena

ARENAS = {
    'raven': RavenEnvAdapter,
    'raven-pixel': RavenPixelEnvAdapter,
    'raven-pixel-ray': RavenPixelEnvAdapterRay,
    'gymnasium': GymnasiumArena
}