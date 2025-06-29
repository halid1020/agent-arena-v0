import logging
from dotmap import DotMap

from ...oracle.rect_fabric.pick_and_place_flattening_policies import *
from ...oracle.rect_fabric.pick_and_place_folding_policies import *


class OracleRectFabricPolicyBuilder():

    ## Example Config String: "oracle_rect_fabric|action:pick-and-place(1),strategy:flattening"
    def build(config_str):
        config = OracleRectFabricPolicyBuilder.parse_config_str(config_str)
        # config['action_space'] = env.get_action_space()
        return OracleRectFabricPolicyBuilder.build_from_config(**config)
    
    def build_from_config(action, task, strategy, num_picker):
        config = {}
        config.update(OracleRectFabricPolicyBuilder.return_config_from_action(action, num_picker))
        config.update(OracleRectFabricPolicyBuilder.return_config_from_strategy(task, strategy))

        print('Agent Config:', config) 

        config = DotMap(config)
        cls =  OracleRectFabricPolicyBuilder.return_class_from_strategy_and_action(task , strategy, action, num_picker)
        return cls(config)

    def return_class_from_strategy_and_action(task, strategy, action, num_picker=2):

        if 'pixel-pick-and-place' in action and strategy in ['oracle-towel', 'noisy-oracle-towel', 'realadapt']:
            
            if 'flattening' == task or 'canonical-flattening' == task:
                if strategy == 'realadapt':
                    from .realadapt_pnp_flattening import RealAdaptPnPFlattening
                    return RealAdaptPnPFlattening
                return OracleTowelPnPFlattening
            
            if 'one-step-folding' == task:
                if num_picker == 1:
                    from agent.oracle.rect_fabric.one_picker_one_step_folding \
                        import OnePickerOneStepFoldingPolicy
                    return OnePickerOneStepFoldingPolicy
                else:
                    raise NotImplementedError
            

            #### Following supports only 1 picker
            if 'diagonal-folding' == task or 'canonical-diagonal-folding' == task:
                from .diagonal_folding_policies \
                    import RectFabricDiagonalFoldingExpertPolicy
                return RectFabricDiagonalFoldingExpertPolicy

            if 'corners-edge-inward-folding' == task:
                from .corners_egde_inward_folding_policies \
                    import RectFabricCornersEdgeFoldingExpertPolicy
                return RectFabricCornersEdgeFoldingExpertPolicy
            
            if 'diagonal-cross-folding' == task or 'canonical-diagonal-cross-folding' == task:
                from .diagonal_cross_folding_policies \
                    import RectFabricDiagonalCrossFoldingExpertPolicy
                return RectFabricDiagonalCrossFoldingExpertPolicy
            
            if 'one-corner-inward-folding' == task or 'canonical-one-corner-inward-folding' == task:
                from .one_corner_inward_folding_policies \
                    import RectFabricOneCornerInwardFoldingExpertPolicy
                return RectFabricOneCornerInwardFoldingExpertPolicy
            
            if 'double-corner-inward-folding' == task or 'canonical-double-corner-inward-folding' == task:
                from .double_corner_inward_folding_policies \
                    import RectFabricDoubleCornerInwardFoldingExpertPolicy
                return RectFabricDoubleCornerInwardFoldingExpertPolicy
            
            if 'all-corner-inward-folding' == task or 'canonical-all-corner-inward-folding' == task:
                from .all_corner_inward_folding_policies \
                    import RectFabricAllCornerInwardFoldingExpertPolicy
                return RectFabricAllCornerInwardFoldingExpertPolicy
            
            

            ### Following supports both 1 and 2 pickers
            if 'side-folding' == task or 'canonical-side-folding' == task:
                if num_picker == 1:
                    #print('here')
                    from agent.oracle.rect_fabric.side_folding_policies \
                        import RectFabricSideFoldingExpertPolicy
                    return RectFabricSideFoldingExpertPolicy
                from agent.oracle.rect_fabric.side_folding_policies \
                    import RectFabricTwoPickerSideFoldingExpertPolicy
                return RectFabricTwoPickerSideFoldingExpertPolicy
        
            # if ('side-folding' == task or 'canonical-side-folding' == task)\
            #       and num_picker == 1:
            #     from agent.oracle.rect_fabric.side_folding_policies \
            #         import RectFabricSideFoldingExpertPolicy
            #     return RectFabricSideFoldingExpertPolicy
            
            if 'double-side-folding' == task or 'canonical-double-side-folding' == task:
                if num_picker == 1:
                    from agent.oracle.rect_fabric.double_side_folding_policies \
                        import RectFabricDoubleSideFoldingExpertPolicy
                    return RectFabricDoubleSideFoldingExpertPolicy
                from agent.oracle.rect_fabric.double_side_folding_policies \
                    import RectFabricTwoPickerDoubleSideFoldingExpertPolicy
                return RectFabricTwoPickerDoubleSideFoldingExpertPolicy
            
            if 'rectangular-folding' == task or 'canonical-rectangular-folding' == task:
                
                if num_picker == 1:
                    #print('here')
                    from .rectangular_folding_policies \
                        import RectFabricRectangularFoldingExpertPolicy
                    return RectFabricRectangularFoldingExpertPolicy
            
                from agent.oracle.rect_fabric.rectangular_folding_policies \
                    import RectFabricTwoPickerRectangularFoldingExpertPolicy
                return RectFabricTwoPickerRectangularFoldingExpertPolicy
            
            
            ## Following does not support so far
            if 'cross-folding' == task or 'canonical-cross-folding' == task:
                from agent.oracle.rect_fabric.cross_folding_policies \
                    import RectFabricTwoPickerCrossFoldingExpertPolicy
                return RectFabricTwoPickerCrossFoldingExpertPolicy
            
            if ('cross-folding' == task or 'canonical-cross-folding' == task)\
                    and num_picker == 1:
                from agent.oracle.rect_fabric.cross_folding_policies \
                    import RectFabricCrossFoldingExpertPolicy
                return RectFabricCrossFoldingExpertPolicy
            
        if 'pixel-pick-and-place' in action and 'expert-flattening-random-folding' == strategy:
            return RectFabricMultiStepFoldingExpertPolicy
        
        if 'pixel-pick-and-place' in action and 'real2sim-smoothing' == strategy:
            from agent_arena.agent.oracle.rect_fabric.realadapt_pnp_flattening \
                import RealAdaptPnPFlattening
            return RealAdaptPnPFlattening

        if 'world-pick-and-place' in action and 'real2sim-smoothing' == strategy:
            from agent_arena.agent.oracle.rect_fabric.realadapt_pnp_flattening \
                import RealAdaptPnPFlattening
            return RealAdaptPnPFlattening

            
           
            
        #### TODO: This does not belong here
        if strategy == 'random':
            return RandomPickAndPlacePolicy
        
        if strategy == 'corner-biased':
            return RectFabricPickAndPlaceCornerBiasedPolicy
        
        if strategy == 'cloth-mask-small-drag':
            return RectFabricClothMaskSmallDragPolicy
        
        if strategy == 'flattening-random-folding':
            return RectFabricMultiStepFoldingExpertPolicy
        
        if 'canonicalise-random-folding-canonicalise' == strategy:
            from agent.oracle.rect_fabric.canonicalise_random_folding_canonicalise \
                import CanonicaliseRandomFoldingCanonicalisePolicy
            return CanonicaliseRandomFoldingCanonicalisePolicy
        
        
        logging.error('[oracle, rect-fabric, builder] strategy {} and task {} do not support'.format(strategy, task))
        raise NotImplementedError

    
    def return_config_from_action(action, num_picker=2):
        config = {}
        if action == 'pixel-pick-and-place-z':
            config['action_dim'] = [2, 6]
            config['action_low'] = [[-1, -1, 0, -1, -1, 0], [-1, -1, 0, -1, -1, 0]]
            config['action_high'] = [[1, 1, 1.5, 1, 1, 1.5], [1, 1, 1.5, 1, 1, 1.5]]
            config['no_op'] = [[1, 1, 0.09, 1, 1, 0.09], [-1, 1, 0.09, -1, 1, 0.09]]
            config['heuristic_pick_z'] = True
            config['heuristic_place_z'] = True
            config['pick_offset'] = 0.02

        elif action == 'pixel-pick-and-place':
            config['action_dim'] = [2, 4]
            config['action_low'] = [[-1, -1, -1, -1], [-1, -1, -1, -1]]
            config['action_high'] = [[1, 1, 1, 1], [1, 1, 1, 1]]
            config['no_op'] = [ [1, 1, 1, 1], [-1, 1, -1, 1]]
        elif action == 'velociy-grasp':
            raise NotImplementedError
        elif action == 'world-pick-and-place':
            config['action_mode'] = 'world-pick-and-place'
            config['action_dim'] = [2, 6]
            config['action_low'] = [[-1, -1, 0, -1, -1, 0], [-1, -1, 0, -1, -1, 0]]
            config['action_high'] = [[1, 1, 1, 1, 1, 1.5], [1, 1, 1, 1, 1, 1]]
            config['no_op'] = [[1, 1, 0.6, 1, 1, 0.6], [-1, 1, 0.6, -1, 1, 0.6]]
        else:
            print("PolicyBuilder: action <{}> does not support".format(action))
            raise NotImplementedError
        
        if num_picker == 1:
            config['action_dim'][0] = 1
            config['action_low'] = [config['action_low'][0]]
            config['action_high'] = [config['action_high'][0]]
            config['no_op'] = [config['no_op'][0]]
        
        return config
    
    def return_config_from_strategy(task, strategy):
        config = {}

        if task == 'flattening' and strategy in ['realadapt', 'oracle-towel']:
            pass
        elif task == 'canonical-flattening':
            config.update({
                'canonical': True
            })
            if strategy == 'canonicalise-random-folding-canonicalise':
                logging.debug('[oracle, rect-fabric, builder] update config for canonicalise-random-folding-canonicalise')
                config.update({
                    'canonical': True,
                    'random_folding_steps': [1, 3],
                    'pick_corner': True,
                    'flatten_noise': False
                })
        elif strategy == 'noisy-oracle-towel' and task == 'flattening':
            config.update({
                'pick_action_noise': 0.05,
                'place_action_noise': 0.05,
                'drag_action_noise': 0.05,
                'flattening_noise': True
            })
        elif 'folding' in task and 'oracle-towel' == strategy:
            config.update({
                'flattening_noise': False,
                'folding_noise': False
            })
            if 'canonical' in task:
                config.update({
                    'canonical': True
                })
        
        elif 'folding' in task and 'realadapt' == strategy:
            print('here!!!')
            config.update({
                'flattening_noise': False,
                'folding_noise': False,
                'real2sim': True
            })
            if 'canonical' in task:
                config.update({
                    'canonical': True
                })
        
        elif 'folding' in task and 'noisy-oracle-towel' == strategy:
            config.update({
                'pick_action_noise': 0.05,
                'place_action_noise': 0.05,
                'drag_action_noise': 0.05,
                'flattening_noise': False,
                'folding_noise': True
            })
               
        elif strategy == 'corner-biased':
            pass
        elif strategy == 'random':
            pass
        elif strategy == 'cloth-mask-small-drag':
             config.update({
                'drag_radius': 0.1
            })
        elif strategy == 'expert-flattening-random-folding':
            config.update({
                'random_folding_steps': [1, 10],
                'flatten_noise': False
            })
        else:
            logging.error('[oracle, rect-fabric, builder]  task <{}> and strategy <{}> does not support'.format(task, strategy))
            raise NotImplementedError
            
        
        return config

    def parse_config_str(config_str):
        config = {}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v
        
        config['num_picker'] = 2
        if '(1)' in config['action']:
            config['num_picker'] = 1
            config['action'] = config['action'].split('(')[0]
        # elif '(2)' in config['action']:
        #     config['num_picker'] = 2
        #     config['action'] = config['action'].split('(')[0]

        return config
