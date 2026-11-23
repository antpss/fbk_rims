import os
import yaml

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def validate_delay_placement(placement):
    if placement != "ex_ante":
        raise ValueError(f"Unsupported delay placement {placement!r}; only 'ex_ante' is supported")

def resolve_path(path):
    """Resolve configured paths consistently, including direct Python callers."""
    return os.path.normpath(os.path.join(PROJECT_ROOT, os.fspath(path))) if path else path


def get_role_mapping_path(cfg):
    models = resolve_path(cfg.get("paths", {}).get("models_dir", "models"))
    return resolve_path(cfg.get("resources", {}).get("discovery", {}).get("output_file")
                        or os.path.join(models, "role_mapping.json"))


def get_profile_paths(cfg):
    models = resolve_path(cfg.get("paths", {}).get("models_dir", "models"))
    profile = cfg.get("activity_profiling", {})
    return (resolve_path(profile.get("output_file") or os.path.join(models, "activity_profiles.json")),
            resolve_path(profile.get("report_file") or os.path.join(models, "activity_profiles_report.md")))


def get_simulation_output_paths(cfg, mode):
    simulation = cfg.get("simulation", {})
    paths = [resolve_path(simulation.get("output_simulated_log", "data/generated/csv/simulated_log.csv")),
             resolve_path(simulation.get("output_simulated_xes", "data/generated/xes/simulated_log.xes"))]
    if mode != "hybrid_residual":
        paths = [f"{os.path.splitext(p)[0]}_{mode}{os.path.splitext(p)[1]}" for p in paths]
    return tuple(paths)


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
        
    try:
        from queue_policy import QueuePolicy
    except ImportError:
        from src.queue_policy import QueuePolicy
    QueuePolicy(config.get("simulation", {}).get("queue")).validate_features(config)

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

