import yaml
import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '.'))
sys.path.append(project_root)
from model_factory import ModelFactory

def load_config(config_path="projects/classify_default.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_model(confi, model_name):
    model = ModelFactory.model_generate(config["training"], model_name)
    model.generate_conf()
    model.generate_data()
    model.generate_model()
    model.train_model()
    # model.export_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument("--config", help="Path to the config file")
    parser.add_argument("--model_type", help="Type of model to be trained")
    parser.add_argument("--model_name", help="Name of the model")
    args = parser.parse_args()
    model_type = args.model_type
    model_name = args.model_name
    if args.config:
        config = load_config(args.config)
    else:
        config_path = f"../projects/{model_type}_default.yaml"
        config = load_config(config_path)
    train_model(config, model_name)
