
# from actoris_harena.agent.drl_algorithms.dreamer_rssm import Dreamer
from actoris_harena.agent.drl.planet.rssm import RSSM
# from actoris_harena.agent.drl_algorithms.planet.rssm_bc import RSSM_BC
# from actoris_harena.agent.drl_algorithms.curl_sac.curl_sac_adapter import CurlSAC_Adapter
# from actoris_harena.agent.drl_algorithms.drq_sac.drq_sac_adapter import DrqSAC_Adapter
# from actoris_harena.agent.drl_algorithms.dreamerV2.dreamer_adapter import DreamerAdapter
# from actoris_harena.agent.drl_algorithms.slac.slac_adapter import SLAC_Adapter
# from actoris_harena.agent.drl_algorithms.planet_conv_gru import ConvRSSM
from actoris_harena.agent.bc.transporter.adapter import TransporterAdapter



from actoris_harena.agent.cloth_control.flatten_then_fold import FlattenThenFold

# from actoris_harena.agent.cloth_control.phase_prediction \
#     import PhasePrediction

from actoris_harena.agent.drl.reinforce import REINFORCE

from actoris_harena.agent.cloth_control.fabricflownet.adapter import FabricFlowNetAdapter
# from actoris_harena.agent.vcd.adapter import VCDAdapter

from actoris_harena.agent.cloth_control.foldsformer.adapter import FoldsformerAdapter
from actoris_harena.agent.cloth_control.cloth_funnel.adapter import ClothFunnel

from actoris_harena.agent.human.pick_and_place.pixel_human_one_picker import PixelHumanOnePicker as PnPHuman1
from actoris_harena.agent.human.pick_and_place.pixel_human_two_picker import PixelHumanTwoPicker as PnPHuman2
from actoris_harena.agent.human.pick_and_fling.pixel_human import PixelHuman as PnFHuman
from actoris_harena.agent.human.pixel_multi_primitive import PixelMultiPrimitive


from actoris_harena.agent.bc.diffusion.adapter import DiffusionAdapter


from actoris_harena.agent.planning.mpc.rect_fabric.pick_and_place_cloth_mask_mpc \
    import RectFabricPickPlaceClothMaskMPC

from actoris_harena.agent.planning.mpc.rect_fabric.pick_and_place_cloth_contour_mpc \
    import RectFabricPickPlaceClothContourMPC

from actoris_harena.agent.planning.mpc.rect_fabric.pick_and_place_cloth_contour_mpc_step_goals \
    import RectFabricPickPlaceClothContourMPCStepGoals

from actoris_harena.agent.random.random_policy import RandomPolicy
from actoris_harena.agent.random.mask_biased_pixel_random_pick_and_place_policy import MaskBiasedPixelPickAndPlacePolicy

from actoris_harena.agent.oracle.raven.raven_oracle_policy_adapter \
    import RavenOraclePolicyAdapter
from actoris_harena.agent.random.raven_mask_biased_random_policy \
    import RavenMaskBiasedRandomPolicy
from actoris_harena.agent.oracle.raven.raven_pixel_oracle_policy_adapter \
    import RavenPixelOraclePolicyAdapter
from actoris_harena.agent.random.raven_pixel_mask_biased_random_policy \
    import RavenPixelMaskBiasedRandomPolicy

AGENTS = {  
    # 'dreamer-planning': Dreamer,
    'planet-clothpick': RSSM,
    'planet': RSSM,
    'diffusion_policy': DiffusionAdapter,
    # 'rssm-bc': RSSM_BC,
    # 'curl_sac': CurlSAC_Adapter,
    # 'drq_sac': DrqSAC_Adapter,
    # 'dreamer': DreamerAdapter,
    # 'slac': SLAC_Adapter,
    # 'planet-conv-gru': ConvRSSM,
    'transporter': TransporterAdapter,
    'ja-tn': TransporterAdapter,
   
    'flatten_then_fold': FlattenThenFold,
    #'phase_prediction':  PhasePredictionactoris_harena.Agent,
    'REINFORCE': REINFORCE,
    'fabricflownet': FabricFlowNetAdapter,
    # 'vcd': VCDAdapter
    'foldsformer': FoldsformerAdapter,

    'rect_fabric_cloth_mask_mpc': RectFabricPickPlaceClothMaskMPC,
    'rect_fabric_cloth_contour_mpc': RectFabricPickPlaceClothContourMPC,
    'rect_fabric_cloth_contour_mpc_step_goals': RectFabricPickPlaceClothContourMPCStepGoals,

    'cloth-funnel': ClothFunnel,

    'raven-oracle': RavenOraclePolicyAdapter,
    'raven-pixel-oracle': RavenPixelOraclePolicyAdapter,
    'raven-mask-biased-random': RavenMaskBiasedRandomPolicy,
    'raven-pixel-mask-biased-random': RavenPixelMaskBiasedRandomPolicy,
    
    'random': RandomPolicy,
    'mask-biased-pixel-random-pick-and-place': MaskBiasedPixelPickAndPlacePolicy,

    'human-pixel-pick-and-place-1': PnPHuman1,
    'human-pixel-pick-and-place-2': PnPHuman2,
    'human-pixel-pick-and-fling': PnFHuman,

    'human-pixel-multi-primitive': PixelMultiPrimitive,
}

# AGENT_NO_CONFIG = {
    
# }


