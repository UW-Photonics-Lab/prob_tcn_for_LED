import optuna
import yaml
import os
from datetime import datetime

STORAGE = "sqlite:///C:/Users/Public_Testing/Desktop/peled_interconnect/mldrivenpeled/experiments/optuna_study.db"

def create_optuna():

    STUDY_NAME = "labview_transformer_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE, direction="minimize")
    with open("study_name.txt", "w") as f:
         f.write(STUDY_NAME)

def choose_hyperparameters():
    ''' Choose hyperparameters for next trainig cycle

        Returns:
            continue: bool determines whether to continue current run
            modulator mode
            epochs: int
            gain: Vpp
            dc_offset: Vdc
    '''
    with open("study_name.txt", "r") as f:
        study_name = f.read().strip()

    study = optuna.load_study(study_name=study_name, storage=STORAGE)
    trial = study.ask()

    with open("trial_number.txt", "w") as f:
        f.write(str(trial.number))

    batch_size = trial.suggest_categorical("batch_size", [1, 8, 16, 32])
    batch_size = 1

    '''
    Modulators:
    m5_apsk_constellation
    qpsk

    '''
    epochs = 250 * batch_size
    scheduler_type = trial.suggest_categorical("scheduler_type", ['reduce_lr_on_plateu'])

    config = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
        "nlayers": trial.suggest_int("nlayers", 2, 6),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512]),
        "batch_size": batch_size,
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "plot_frequency": 10 * batch_size,
        "save_model_frequency": 500,
        # "EARLY_STOP_PATIENCE": 500 // batch_size, 
        "EARLY_STOP_PATIENCE": 1,
        "EARLY_STOP_THRESHOLD": 0.5,
        "modulator": 'm5_apsk_constellation',
        "epochs": 250 * batch_size,
        "gain" : 20,
        "dc_offset": 0,
        "optuna_study": study_name,
        "num_symbols_per_frame" : 1,
        "scheduler_type": scheduler_type,
        # "weight_init": trial.suggest_categorical("weight_init", ["xavier", "kaiming", "normal", "default"]),
        "weight_init": "default",
        "CP_ratio": 0.25,
        "channel_derivative_type": trial.suggest_categorical("channel_derivative_type", ["linear", "ici_matrix"]),
        # "pre_layer_norm": trial.suggest_categorical("pre_layer_norm", [True, False]),
        "pre_layer_norm": False,
        "ici_window_length": 370
    }

    if config["scheduler_type"] == "warmup":
         config["warmup_steps"] = trial.suggest_int("warmup_steps", 0, int(0.2 * epochs))

    if config["channel_derivative_type"] == "ici_matrix":
         config['matrix_regularization'] = trial.suggest_float("matrix_regularization", 1e-6, 1e-3, log=True)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.yml")
    # Write to config file for LabVIEW
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config['modulator'], config['epochs'], config['gain'], config['dc_offset'], config['num_symbols_per_frame']

def report_result():
    with open("study_name.txt", "r") as f:
        study_name = f.read().strip()

    study = optuna.load_study(study_name=study_name, storage=STORAGE)

    with open("trial_number.txt", "r") as f:
            trial_number = int(f.read())


    with open("final_loss.txt", "r") as f:
        loss_value = float(f.read())

    study.tell(trial_number, loss_value)