import pickle
from typing import Any


def pickel_loader(flie_name: str) -> Any:
    """
    This function loads and returns the data from a pickle file.

    Parameters:
    flie_name (str): The name of the pickle file to be loaded.

    Returns:
    Any: The data loaded from the pickle file.
    """
    with open(flie_name, "rb") as f:
        return pickle.load(f)
