import optuna
import yaml
import os

STORAGE = "sqlite:///optuna_study.db"
STUDY_NAME = "labview_transformer"

def choose_hyperparameters():
    ''' Choose hyperparameters for next trainig cycle

        Returns:
            continue: bool determines whether to continue current run
            modulator mode
            epochs: int
            gain: Vpp
            dc_offset: Vdc

    '''
    
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except KeyError:
        study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE, direction="minimize")
    trial = study.ask()

    batch_size = trial.suggest_categorical("batch_size", [16, 32])


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
        "modulator": 'm5_apsk_constellation',
        "epochs": 10000,
        "gain" : 20,
        "dc_offset": 0
    }


    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.yml")
    # Write to config file for LabVIEW
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Save the trial ID so LabVIEW can report results later
    with open("trial_id.txt", "w") as f:
        f.write(str(trial._trial_id))

    return config['modulator'], config['epochs'], config['gain'], config['dc_offset']

def report_result():
    study = optuna.load_study(study_name="labview_transformer", storage="sqlite:///optuna_study.db")

    with open("trial_id.txt", "r") as f:
        trial_id = int(f.read())

    with open("final_loss.txt", "r") as f:
        loss_value = float(f.read())

    trial = study.get_trial(trial_id)
    study.tell(trial, loss_value)