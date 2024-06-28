from pathlib import Path

def get_config():
    return {
        'batch_size': 8,
        'num_epochs': 10,
        'lr': 1e-5,
        'd_model': 512,
        'seq_length': 350,
        'lang_src': 'en',
        'lang_target': 'it',
        'model_folder': "weights",
        "model_filename": "tmodel_",
        'preload': None,
        'tokenizer_file': "tokenizer_{0}.json",
        'experiment_name': "runs/tmodel"
    }


def get_weights_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
