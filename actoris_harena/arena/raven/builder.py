from .raven_env_adapter import RavenEnvAdapter
from dotmap import DotMap

class RavenBuilder():

    @staticmethod
    def build(config_str):
        # Parses string into dict, e.g., {'task': '...', 'img_res': '128'}
        config = RavenBuilder.parse_config_str(config_str)
        # Unpacks dict into keyword arguments
        return RavenBuilder.build_from_config(**config)
    
    @staticmethod
    def build_from_config(task, disp, **kwargs):
        """
        Builds the environment adapter.
        
        Args:
            task: The task name.
            disp: Display flag (bool or string).
            **kwargs: Catches extra arguments like 'img_res', 'view_mode'.
        """
        
        # 1. Robust Boolean Conversion
        # Handles "True", "true", True (bool), or 1 (int)
        if isinstance(disp, str):
            disp = disp.lower() == 'true'
        else:
            disp = bool(disp)
            
        # 2. Create the base configuration dictionary
        config_dict = {
            'task': task,
            'disp': disp
        }
        
        # 3. CRITICAL FIX: Merge the extra arguments (kwargs) into the config
        # This ensures 'img_res' and 'view_mode' are added to the DotMap
        config_dict.update(kwargs)

        config = DotMap(config_dict)
        
        # Now RavenEnvAdapter receives the full config
        env = RavenEnvAdapter(config)
        return env

    @staticmethod
    def parse_config_str(config_str):
        config = {}
        
        # Basic error handling for string format
        parts = config_str.split('|')
        if len(parts) < 2:
            print(f"Warning: Config string '{config_str}' format might be incorrect. Expected 'name|key:val,...'")
            params = parts[0] # Fallback if no pipe exists
        else:
            params = parts[1]

        items = params.split(',')

        for i in items:
            if ':' in i:
                k, v = i.split(':')
                config[k] = v
            
        return config