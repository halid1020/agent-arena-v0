from agent_arena.utilities.logger.rect_fabric.pick_and_place_rect_fabric_single_task_logger \
    import PickAndPlaceRectFabricSingleTaskLogger

class PickAndPlaceRectFabricAllTaskLogger(PickAndPlaceRectFabricSingleTaskLogger):
    def __init__(self, log_dir):
        super().__init__(log_dir)
    
    def __call__(self, episode_config, result):

        ## Call Super
        super().__call__(episode_config, result, filename=episode_config['task'])
    
    def check_exist(self, episode_config):
        ## Call Super
        return super().check_exist(episode_config, filename=episode_config['task'])
