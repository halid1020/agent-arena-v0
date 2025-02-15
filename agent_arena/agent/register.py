
# from agent_arena.agent.drl_algorithms.dreamer_rssm import Dreamer
from agent_arena.agent.drl.planet.rssm import RSSM
# from agent_arena.agent.drl_algorithms.planet.rssm_bc import RSSM_BC
# from agent_arena.agent.drl_algorithms.curl_sac.curl_sac_adapter import CurlSAC_Adapter
# from agent_arena.agent.drl_algorithms.drq_sac.drq_sac_adapter import DrqSAC_Adapter
# from agent_arena.agent.drl_algorithms.dreamerV2.dreamer_adapter import DreamerAdapter
# from agent_arena.agent.drl_algorithms.slac.slac_adapter import SLAC_Adapter
# from agent_arena.agent.drl_algorithms.planet_conv_gru import ConvRSSM
from agent_arena.agent.bc.transporter.adapter import TransporterAdapter



from agent_arena.agent.flatten_then_fold import FlattenThenFold

# from agent_arena.agent.cloth_control.phase_prediction \
#     import PhasePrediction

from agent_arena.agent.reinforce import REINFORCE

from agent_arena.agent.cloth_control.fabricflownet.adapter import FabricFlowNetAdapter
# from agent_arena.agent.vcd.adapter import VCDAdapter

from agent_arena.agent.cloth_control.foldsformer.adapter import FoldsformerAdapter
from agent_arena.agent.cloth_control.cloth_funnel.adapter import ClothFunnel

from agent_arena.agent.human.pick_and_place.pixel_human_one_picker import PixelHumanOnePicker as PnPHuman1
from agent_arena.agent.human.pick_and_place.pixel_human_two_picker import PixelHumanTwoPicker as PnPHuman2
from agent_arena.agent.human.pick_and_fling.pixel_human import PixelHuman as PnFHuman
from agent_arena.agent.human.pixel_multi_primitive import PixelMultiPrimitive


from agent_arena.agent.diffusion.adapter import DiffusionAdapter


from agent_arena.agent.planning.mpc.rect_fabric.pick_and_place_cloth_mask_mpc \
    import RectFabricPickPlaceClothMaskMPC

from agent_arena.agent.planning.mpc.rect_fabric.pick_and_place_cloth_contour_mpc \
    import RectFabricPickPlaceClothContourMPC

from agent_arena.agent.planning.mpc.rect_fabric.pick_and_place_cloth_contour_mpc_step_goals \
    import RectFabricPickPlaceClothContourMPCStepGoals

from agent_arena.agent.random_policy import RandomPolicy

AGENT_NEEDS_CONFIG = {  
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
    #'phase_prediction':  PhasePredictionagent_arena.Agent,
    'REINFORCE': REINFORCE,
    'fabricflownet': FabricFlowNetAdapter,
    # 'vcd': VCDAdapter
    'foldsformer': FoldsformerAdapter,

    'rect_fabric_cloth_mask_mpc': RectFabricPickPlaceClothMaskMPC,
    'rect_fabric_cloth_contour_mpc': RectFabricPickPlaceClothContourMPC,
    'rect_fabric_cloth_contour_mpc_step_goals': RectFabricPickPlaceClothContourMPCStepGoals,

    'cloth-funnel': ClothFunnel
    
}

AGENT_NO_CONFIG = {
    'random': RandomPolicy,
    'human-pixel-pick-and-place-1': PnPHuman1,
    'human-pixel-pick-and-place-2': PnPHuman2,
    'human-pixel-pick-and-fling': PnFHuman,

    'human-pixel-multi-primitive': PixelMultiPrimitive,
}



