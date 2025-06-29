from agent_arena.agent.agent import Agent


class FlattenThenFold(Agent):
    
        def __init__(self, config):
            self.config = config
            self.agent_name = "transporter"
            
            import agent_arena.api as ag_ar
            flatten_config = ag_ar.retrieve_config(
                config.flatten_agent.name,
                config.flatten_agent.arena,
                config.flatten_agent.config,
                config.flatten_agent.log_dir)
            self.flatten_agent = ag_ar.build_agent(self.agent_name, config=flatten_config)
            
            fold_config = ag_ar.retrieve_config(
                config.fold_agent.name,
                config.fold_agent.arena,
                config.fold_agent.config,
                config.fold_agent.log_dir)
            self.fold_agent = ag_ar.build_agent(self.agent_name, config=fold_config)

            self.flatten_phase = True
            self.name = self.flatten_agent.get_name() + "_" + self.fold_agent.get_name()

        def act(self, state):
            if state['normalised_coverage'] >= self.config.coverage_threshold:
                self.flatten_phase = False
            if self.flatten_phase:
                return self.flatten_agent.act(state)
            else:
                return self.fold_agent.act(state)
            
        def get_phase(self):
            if self.flatten_phase:
                return "flattening"
            else:
                return "folding"
        
        def success(self):
            return self.fold_agent.success()
        

        def reset(self):
            self.flatten_phase = True
        
        def init(self, state):
            self.flatten_agent.init(state)
            self.fold_agent.init(state)
        
        def update(self, state, action):
            self.flatten_agent.update(state, action)
            self.fold_agent.update(state, action)
    