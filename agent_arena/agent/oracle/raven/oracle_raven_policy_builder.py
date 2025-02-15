from agent_arena.agent.oracle.raven.oracle_raven_policy_wrapper \
    import OracleRavenPolicyWrapper
from dotmap import DotMap

class OracleRavenPolicyBuilder():

    def build(config_str):
        config = OracleRavenPolicyBuilder.parse_config_str(config_str)
        config = DotMap(config)
        return OracleRavenPolicyBuilder.build_from_config(config)
    
    def build_from_config(config):
        return OracleRavenPolicyWrapper(config)
    
    def parse_config_str(config_str):
        config = {}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v
        return config