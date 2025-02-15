from .border_biased_pixel_pick_and_place_policy \
    import GarmentBorderBiasedPixelPickAndPlacePolicy
from .mask_biased_pixel_pick_and_place_policy \
    import GarmentMaskBiasedPixelPickAndPlacePolicy

class OracleGarmentPolicyBuilder():

    ## Example Config String: "oracle_rect_fabric|action:pick-and-place(1),strategy:expert-flattening"
    def build(config_str):
        config = OracleGarmentPolicyBuilder.parse_config_str(config_str)
        return OracleGarmentPolicyBuilder.build_from_config(**config)
    
    def build_from_config(policy_name):
        return  OracleGarmentPolicyBuilder.return_class_from_strategy_and_action(policy_name)
    
    def return_class_from_strategy_and_action(policy_name):
        if 'border-biased-pick-and-place' == policy_name:
            return GarmentBorderBiasedPixelPickAndPlacePolicy(config=None)
        elif 'mask-biased-pick-and-place' == policy_name:
            return GarmentMaskBiasedPixelPickAndPlacePolicy(config=None)
        else:
            raise NotImplementedError

    def parse_config_str(config_str):
        config = {'policy_name': config_str.split('|')[1]}
        return config
