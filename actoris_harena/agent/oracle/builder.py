import logging

class OracleBuilder():

    def build(config_str, param={}):
        
        if config_str == 'rect_fabric_mpc_readjust_pick':
            from ..oracle.rect_fabric.pick_and_place_readjust_mpc \
                import RectFabricPickPlaceReadjustMPC
            param['model'] = param['base_policy']
            return  RectFabricPickPlaceReadjustMPC(**param)
        
        if config_str == 'max_action':
            from ..oracle.base_policies import MaxActionPolicy
            param['action_space'] = arena.get_action_space()
            return MaxActionPolicy(**param)
        
        if config_str == 'rect_fabric_cloth_mask_mpc':
            from ..oracle.rect_fabric.pick_and_place_cloth_mask_mpc \
                import RectFabricPickPlaceClothMaskMPC
            param['model'] = param['base_policy']
            param['action_space'] = arena.get_action_space()
            return  RectFabricPickPlaceClothMaskMPC(**param)
        
        if config_str == 'rect_fabric_cloth_contour_mpc':
            from ..oracle.rect_fabric.pick_and_place_cloth_contour_mpc \
                import RectFabricPickPlaceClothContourMPC
            param['model'] = param['base_policy']
            param['action_space'] = arena.get_action_space()
            return  RectFabricPickPlaceClothContourMPC(**param)
        
        if config_str == 'rect_fabric_cloth_edge_mpc':
            from ..oracle.rect_fabric.pick_and_place_cloth_edge_mpc \
                import RectFabricPickPlaceClothEdgeMPC
            param['model'] = param['base_policy']
            param['action_space'] = arena.get_action_space()
            return  RectFabricPickPlaceClothEdgeMPC(**param)
        
        if config_str == 'rect_fabric_cloth_contour_mpc_step_goals':
            from ..oracle.rect_fabric.pick_and_place_cloth_contour_mpc_step_goals \
                import RectFabricPickPlaceClothContourMPCStepGoals
            param['model'] = param['base_policy']
            param['action_space'] = arena.get_action_space()
            return  RectFabricPickPlaceClothContourMPCStepGoals(**param)
        
        if config_str == 'rect_fabric_wrinkels':
            from ..oracle.rect_fabric.wrinkels_policy \
                import WrinklesPolicy
            param['action_space'] = arena.get_action_space()
            return  WrinklesPolicy(**param)
        
        if config_str == 'mpc_cem':
            from ..oracle.mpc_cem import MPC_CEM
            param['model'] = param['base_policy']
            param['action_space'] = arena.get_action_space()
            return MPC_CEM(**param)
        
        if config_str == 'success-no-op':
            from ..oracle.success_no_op import SuccessNoOp
            return SuccessNoOp(**param)
    
        if config_str == 'no-op':
            from ..oracle.no_op import NoOp
            return NoOp(param)
        
        if config_str == 'visual_mpc_cem':
            from ..oracle.visual_mpc_cem import VisualMPC_CEM
            param['model'] = param['.']
            return VisualMPC_CEM(param)
        
        if config_str == 'random':
            from ..random_policy import RandomPolicy
            return RandomPolicy()

        if config_str == 'goal_action':
            from ..oracle.goal_action_policy import GoalActionPolicy
            return GoalActionPolicy()

        target_builder = config_str.split('|')[0]
        if target_builder == 'oracle-rect-fabric':
            from ..oracle.rect_fabric.builder \
                import OracleRectFabricPolicyBuilder
            return OracleRectFabricPolicyBuilder.build(config_str)
        
        elif target_builder == 'oracle-garment':

            from ..oracle.garment.oracle_garment_policy_builder\
                  import OracleGarmentPolicyBuilder
            return OracleGarmentPolicyBuilder.build(config_str)
        
        elif target_builder == 'raven':
            from ..oracle.raven.oracle_raven_policy_builder \
                import OracleRavenPolicyBuilder
            return OracleRavenPolicyBuilder.build(config_str)
        
        elif target_builder == 'deformable-raven':
            from ..oracle.deformable_raven.oracle_policy_builder \
                import OraclePolicyBuilder
            return OraclePolicyBuilder.build(config_str)
        
        else:
            logging.error("[oracle builder] It does not support bulding from <{}> "\
                  .format(target_builder))
            raise NotImplementedError