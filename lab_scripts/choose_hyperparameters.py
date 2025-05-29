import optuna
import yaml
import os
from datetime import datetime

STORAGE = "sqlite:///C:/Users/Public_Testing/Desktop/peled_interconnect/mldrivenpeled/experiments/optuna_study.db"

def create_optuna():

    STUDY_NAME = "labview_transformer_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE, direction="minimize")
    # trial = study.ask()
    # trial_number = trial.number
    # with open("trial_number.txt", "w") as f:
    #     f.write(str(trial_number))

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

    # batch_size = trial.suggest_categorical("batch_size", [1, 4, 16, 32])
    batch_size = 1


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
        "EARLY_STOP_PATIENCE": 2000, 
        "EARLY_STOP_THRESHOLD": 0.5,
        "modulator": 'qpsk',
        "epochs": 2,
        "gain" : 20,
        "dc_offset": 0,
        "optuna_study": study_name
    }


    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.yml")
    # Write to config file for LabVIEW
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config['modulator'], config['epochs'], config['gain'], config['dc_offset']

def report_result():
    with open("study_name.txt", "r") as f:
        study_name = f.read().strip()

    study = optuna.load_study(study_name=study_name, storage=STORAGE)

    with open("trial_number.txt", "r") as f:
            trial_number = int(f.read())


    with open("final_loss.txt", "r") as f:
        loss_value = float(f.read())

    study.tell(trial_number, loss_value)