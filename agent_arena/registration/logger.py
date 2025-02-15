from agent_arena.utilities.loggers import *

from agent_arena.utilities.logger.dummy_logger import DummyLogger
from agent_arena.utilities.logger.rect_fabric.pick_and_place_rect_fabric_single_task_logger \
    import PickAndPlaceRectFabricSingleTaskLogger
from agent_arena.utilities.logger.rect_fabric.pick_and_place_rect_fabric_all_task_logger \
    import PickAndPlaceRectFabricAllTaskLogger
from agent_arena.utilities.logger.standard_logger import StandardLogger
from agent_arena.utilities.logger.rect_fabric.pick_and_place_phase_prediction_logger \
    import PickAndPlacePhasePredictionLogger

LOGGER = {
    'dummy_logger': DummyLogger,
    'pick_and_place_fabric_single_task_logger':
        PickAndPlaceRectFabricSingleTaskLogger,
    'standard_logger': StandardLogger,
    'save_goal_logger': save_goal_logger,
    'pick_and_place_rect_fabric_all_task_manupilation_logger': \
        PickAndPlaceRectFabricAllTaskLogger,
    'pick_and_place_phase_prediction_logger': \
        PickAndPlacePhasePredictionLogger,
}