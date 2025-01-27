import yaml


def load_config(file_path: str = None) -> dict:
    """
    Load the configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data as a dictionary.

    Raises:
        ValueError: If the file_path is not provided.
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if file_path is None:
        raise ValueError("Configuration file path must be provided. None was given.")

    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found at path: {file_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file at path: {file_path}") from e
