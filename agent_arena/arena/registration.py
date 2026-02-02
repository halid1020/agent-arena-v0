from .raven.raven_env_adapter import RavenEnvAdapter
from .raven.raven_pixel_env_adapter import RavenPixelEnvAdapter

ARENAS = {
    'raven': RavenEnvAdapter,
    'raven-pixel': RavenPixelEnvAdapter
}