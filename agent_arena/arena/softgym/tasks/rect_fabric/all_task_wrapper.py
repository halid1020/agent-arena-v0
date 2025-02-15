from arena.softgym.task_wrappers.task_wrapper import TaskWrapper

# TODO: put the following into a spereate file
from arena.softgym.task_wrappers.rect_fabric.all_corner_inward_folding_wrapper \
    import AllCornerInwardFoldingWrapper
from arena.softgym.task_wrappers.rect_fabric.diagonal_folding_wrapper \
    import DiagonalFoldingWrapper
from arena.softgym.task_wrappers.rect_fabric.diagonal_cross_folding_wrapper \
    import DiagonalCrossFoldingWrapper
from arena.softgym.task_wrappers.rect_fabric.cross_folding_wrapper \
    import CrossFoldingWrapper
from arena.softgym.task_wrappers.rect_fabric.double_corner_inward_folding_wrapper \
    import DoubleCornerInwardFoldingWrapper
from arena.softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
    import DoubleSideFoldingWrapper
from arena.softgym.task_wrappers.rect_fabric.flattening_wrapper \
    import FlatteningWrapper
from arena.softgym.task_wrappers.rect_fabric.one_corner_inward_folding_wrapper \
    import OneCornerInwardFoldingWrapper
from arena.softgym.task_wrappers.rect_fabric.rectangular_folding_wrapper \
    import RectangularFoldingWrapper
from arena.softgym.task_wrappers.rect_fabric.side_folding_wrapper \
    import SideFoldingWrapper


TASK_WRAPPER = {
    'all-corner-inward-folding': AllCornerInwardFoldingWrapper,
    # 'cross-folding': CrossFoldingWrapper,
    'diagonal-cross-folding': DiagonalCrossFoldingWrapper,
    'diagonal-folding': DiagonalFoldingWrapper,
    'double-corner-inward-folding': DoubleCornerInwardFoldingWrapper,
    'double-side-folding': DoubleSideFoldingWrapper,
    'flattening': FlatteningWrapper,
    'one-corner-inward-folding': OneCornerInwardFoldingWrapper,
    'rectangular_folding': RectangularFoldingWrapper,
    'side-folding': SideFoldingWrapper,
}

class AllTaskWrapper(TaskWrapper):
    def __init__(self, env):
        self.env = env
        self.base_env = env
        self.eval_params = []

        for task in TASK_WRAPPER.keys():
            self.eval_params.extend([
                {
                    'eid': eid,
                    'tier': tier,
                    'save_video': (eid in self.eval_para['video_episodes']),
                    'task': task
                } for tier in self.eval_para['eval_tiers'] \
                    for eid in self.eval_para['eval_tiers'][tier]
            ])
        #self.reset()

    def reset(self, episode_config={'eid': None, 'task': 'flattening'}):
        self.env = TASK_WRAPPER[episode_config['task']](self.base_env)
        return self.env.reset(episode_config)
    
    def evaluate(self):
        return self.env.evaluate()