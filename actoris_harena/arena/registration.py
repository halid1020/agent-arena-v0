from .raven.raven_env_adapter import RavenEnvAdapter
from .raven.raven_pixel_env_adapter import RavenPixelEnvAdapter
from .openAI_gym.wrapper import OpenAIGymArena

ARENAS = {
    'raven': RavenEnvAdapter,
    'raven-pixel': RavenPixelEnvAdapter,
    'openAI-gym': OpenAIGymArena
}