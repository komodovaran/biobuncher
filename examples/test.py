from glob import glob
def _get_latest_model(model_dir):
    """
    Returns most recent model
    """
    model_dir += "/model_???"


d = _get_latest_model("./logs/20191104-195104/")