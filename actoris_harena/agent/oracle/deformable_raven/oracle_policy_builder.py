from agent.oracle.deformable_raven.oracle_policy_wrapper \
    import OraclePolicyWrapper

class OraclePolicyBuilder():

    def build(config_str, env):
        config = OraclePolicyBuilder.parse_config_str(config_str)
        return OraclePolicyBuilder.build_from_config(config['task'], env)
    
    def build_from_config(task, env):
        return OraclePolicyWrapper(task, env)
    
    def parse_config_str(config_str):
        config = {}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v
        return config