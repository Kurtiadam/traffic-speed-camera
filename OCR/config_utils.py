import yaml
import os

def load_config(yaml_file = "ocr_config.yaml"):
    """
    Loads config yaml file. File contect can be accessed as a nested dictionary, e.g. config["data"]["batch_size"].
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file_path = os.path.join(current_dir, yaml_file)

    with open(yaml_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config