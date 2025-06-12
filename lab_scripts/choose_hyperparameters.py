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

    '''
    Modulators:
    m5_apsk_constellation
    qpsk

    '''
    epochs = 150 * batch_size
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    warmup_steps = int(warmup_ratio * epochs)
    # scheduler_type = trial.suggest_categorical("scheduler_type", ['warmup', 'reduce_lr_on_plateu'])
    scheduler_type = "reduce_lr_on_plateu"

    # weight_init = trial.suggest_categorical("weight_init", ["xavier", "kaiming", "normal", "default"])
    weight_init = "default"

    config = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
        "nlayers": trial.suggest_int("nlayers", 2, 6),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 512]),
        "batch_size": batch_size,
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "plot_frequency": batch_size,
        "save_model_frequency": 500,
        "EARLY_STOP_PATIENCE": 100 // batch_size, 
        "EARLY_STOP_THRESHOLD": 0.5,
        "modulator": 'm5_apsk_constellation',
        "epochs": epochs,
        "gain" : 20,
        "dc_offset": 0,
        "optuna_study": study_name,
        "num_symbols_per_frame" : 1,
        "scheduler_type": scheduler_type,
        "warmup_steps": warmup_steps,
        "weight_init": weight_init,
        "f3dB": 6e6,
        "OFDM_period": 1e4,
        "num_points_time": 16384,
        # "cnn": trial.suggest_categorical("cnn", [True, False])
        "cnn": False
    }


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