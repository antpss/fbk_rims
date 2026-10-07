import os
import yaml

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def load_config(config_path=None):
    """
    Loads YAML configuration from config_path or default root config.yaml.
    Resolves relative paths relative to the project root.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Helper to resolve project-relative paths
    def resolve_path(rel_or_abs):
        if not rel_or_abs:
            return rel_or_abs
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.normpath(os.path.join(PROJECT_ROOT, rel_or_abs))
        
    # Pre-resolve all path keys for convenience
    if "paths" in config and isinstance(config["paths"], dict):
        for k in list(config["paths"].keys()):
            if isinstance(config["paths"][k], str):
                config["paths"][k] = resolve_path(config["paths"][k])

    if "preprocessing" in config and isinstance(config["preprocessing"], dict):
        if "input_log" in config["preprocessing"] and isinstance(config["preprocessing"]["input_log"], str):
            config["preprocessing"]["input_log"] = resolve_path(config["preprocessing"]["input_log"])
                
    return config

if __name__ == "__main__":
    cfg = load_config()
    print("Loaded configuration successfully!")
    print(f"Project Name: {cfg['project']['name']}")
    print(f"Aligned Log Path: {cfg['paths']['aligned_log']}")
    print(f"Available Tasks: {list(cfg['tasks'].keys())}")

