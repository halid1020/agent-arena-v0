from agent_arena.utilities.transform.pick_and_place_transformer \
    import PickAndPlaceTransformer
from agent_arena.utilities.transform.pick_and_place_heatmap_transformer \
    import PickAndPlaceHeatmapTransformer
from agent_arena.utilities.transform.transporter_net_transformer \
    import TransporterNetTransformer
from agent_arena.utilities.transform.transporter_net_goal_condition_transformer \
    import TransporterNetGoalConditionTransformer
from agent_arena.utilities.transform.identity_transformer \
    import IdentityTransformer
from agent_arena.utilities.transform.contrastive_learning_transformer \
    import ContrastiveLearningTransformer
from agent_arena.utilities.transform.planet_transformer \
    import PlaNetTransformer
from agent_arena.utilities.transform.phase_prediction_transformer \
    import PhasePredictionTransformer

DATA_TRANSFORMER = {
    'planet_transformer': PlaNetTransformer,
    'pick_and_place_transformer':  PickAndPlaceTransformer,
    'pick_and_place_heatmap_transformer': PickAndPlaceHeatmapTransformer,
    'transporter_net_transformer': TransporterNetTransformer,
    'transporter_net_goal_condition_transformer': TransporterNetGoalConditionTransformer,
    'identity': IdentityTransformer,
    'contrastive_learning_transformer': ContrastiveLearningTransformer,
    'phase_prediction_transform': PhasePredictionTransformer,
}