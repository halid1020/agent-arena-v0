class ArenaBuilder:
    """
    A builder class that dispatches to specific implementation builders 
    based on the configuration string.
    """

    @staticmethod
    def _build_softgym(config_str, ray):
        from .softgym.builder import SoftGymBuilder
        return SoftGymBuilder.build(config_str, ray=ray)

    @staticmethod
    def _build_raven(config_str, ray):
        from .raven.builder import RavenBuilder
        return RavenBuilder.build(config_str)

    @staticmethod
    def _build_deformable_raven(config_str, ray):
        # Note: Preserving original typos 'bulider' and 'Deformbale' 
        # to ensure it matches your file structure.
        from .deformable_raven.bulider import DeformbaleRavenBuilder
        return DeformbaleRavenBuilder.build(config_str)

    @staticmethod
    def _build_dm_control(config_str, ray):
        from .dm_control.builder import DM_ControlBuilder
        return DM_ControlBuilder.build(config_str)

    @staticmethod
    def _build_openai(config_str, ray):
        from .openAI_gym.builder import OpenAIGymBuilder
        return OpenAIGymBuilder.build(config_str)

    # 1. The Dispatch Dictionary
    # We map the string keys to the functions defined above.
    _BUILDER_MAP = {
        'softgym': _build_softgym,
        'raven': _build_raven,
        'deformable-raven': _build_deformable_raven,
        'dm-control-suite': _build_dm_control,
        'openAI-gym': _build_openai,
    }

    @classmethod
    def build(cls, config_str, ray=False):
        target_builder = config_str.split('|')[0]

        # 2. The Lookup
        builder_func = cls._BUILDER_MAP.get(target_builder)

        if builder_func is None:
            print(f"EnvBuilder: target_builder <{target_builder}> is not supported")
            raise NotImplementedError(f"Builder '{target_builder}' not found.")

        # 3. The Execution
        return builder_func(config_str, ray)