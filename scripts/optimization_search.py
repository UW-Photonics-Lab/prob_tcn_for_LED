'''
Authors: Dylan Jones

DEPENDENCIES:
    1. all_waves_f3dB_experiment.csv from Google Drive placed in a directory
       /data/training/prototype/ created manually. All data files are not tracked by Git because
       of memory constraints.

    2. The optimization process takes an initially trained pickled model that must
        be manually named and put into /models/pickled_models/current_best_model.pth.
        This is to separate the models automatically created by initial training functionality,
        so that one can only have desirable models in the optimization process

    3. The optimization process automatically creates large plot folders to store
       itemized results. You need to create a folder ./plots/optimization_plots
       for the results to be stored (also not tracked by Git)

This file achieves two goals:
    1.  If train_initial is set to True, it can create a pickle file of a variable_cvae model to be used in
        the optimization process. Think of this model as the "initial start" as the optimization process
        will always start with this training state.

    2. An optimization process whereby an initially trained model is put into an "experimental simulation"
       where it takes an initial dataset (possibly all_waves_csv) and creates an new optimization_data_set.pkl
       from the original training set. CRITICAL INFORMATION: this data set if undisturbed will accumulate
       data from all previous optimization trials (analogous to real experiments) as the data is just a
       mapping from waveform -> f3dB for the Naive circuit model and is thus agnostic to the model that
       generated it.

       The optimization process proceeds as follows (and has optimization hyperparameters):
            1. Use existing optimization dataset or create new one based on original
               training set if unavailable

            2. Use the configured "prompt_f3dB" to generate waves hoping to
                achieve such f3dB. The number of sample waves is configurable

            3. Use the Naive circuit model to assign lables to all of these samples.
               This is by far the slowest part of the code!

            3. OPTIONAL: filter the new data to add the top X% performing samples

            4. Concatenate new data to optimization_data_set

            5. Performing a number of training/val epochs on new dataset

            6. Create large amount of plots to track model performance. Then, set prompt
               f3dB higher in hopes the model gradually learns higher performing mappings
               between wave -> f3dB.

            REPEAT AT STEP 2 UNTIL ALL CYCLES FINISHED (also configurable)
'''

import torch
import random
import sys
import os
import torch.utils.data
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
import pickle
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from variable_cvae_model import CVAE
from data_generator import *


if (torch.cuda.is_available()):
    device = torch.device("cuda")
elif (torch.backends.mps.is_available()):
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Starting Optimization on device:", device, "\n")

path_to_root_csv = "../data/training/prototype/all_waves_f3dB_experiment.csv"
df = pd.read_csv(path_to_root_csv, index_col=0)

labels = df['f3db'].to_numpy()

voltages = df.iloc[:, 1:]
num_voltages = len(voltages.columns)
waves = []
for _, row in voltages.iterrows():
    waves.append(row.to_numpy())


# 10% Test 10% Val 80% Train
X_train, X_test, y_train, y_test = train_test_split(waves, labels, test_size=0.2, random_state=42)
X_val, X_test, y_yal, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
print("Number of training points:", len(X_train))
print("Test/Validation Size:", len(X_test))

train_initial = True # Set true if you want to retrain model before optimization
batch_size = 64
latent_size = 71
epochs = 1
beta_factor = 0.1
learning_rate = 0.003
max_grad_norm = 1e3
useBatchNorm = True
max_channel_size = 64
kernels = [151, 121, 35]

class OptimizationDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_labels = self.labels[idx]
        if self.transform:
            sample_data = self.transform(sample_data)
            sample_labels = self.transform(sample_labels)

        return sample_data, sample_labels

    def _get_data(self):
        return self.data, self.labels

    def add_data(self, new_data, new_labels):
        self.data = np.concatenate((self.data, new_data), axis=0)
        self.labels = np.concatenate((self.labels, new_labels), axis=0)

    def __len__(self):
        return len(self.data)

train_data_set = OptimizationDataset(X_train, y_train, transform=lambda x: x.astype('float32'))
train_loader = torch.utils.data.DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)

test_data_set = OptimizationDataset(X_test, y_test, transform=lambda x: x.astype('float32'))
test_loader = torch.utils.data.DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=True)

val_data_set = OptimizationDataset(X_val, y_yal, transform=lambda x: x.astype('float32'))
val_loader = torch.utils.data.DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=True)

model = CVAE(num_voltages,
             latent_size=latent_size,
             kernel_sizes=kernels,
             use_batch_norm=True,
             channel_size=max_channel_size).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

def loss_function(recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + beta_factor * KLD

# Handy function that stops everything if numbers blow up or
# become undefined
torch.autograd.set_detect_anomaly(True)
def train(epoch,
          train_losses,
          optimizer_reference,
          loader_reference,
          model,
          make_plots=True, # Prints out reconstruction plots
          scheduler_reference=None):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(loader_reference):

        data, labels = data.to(device), labels.to(device)
        recon_batch, mu, logvar = model(data, labels)

        optimizer_reference.zero_grad()

        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)

        train_loss += loss.detach().cpu().numpy()
        optimizer_reference.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_reference.dataset),
                100. * batch_idx / len(loader_reference),
                loss.item() / len(data)))

        if scheduler_reference is not None:
            scheduler_reference.step()

    tot_train_loss = train_loss / len(loader_reference.dataset)
    train_losses.append(tot_train_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, tot_train_loss))
    print(f"Reconstruction of wave with f3db {labels[0]}")
    if make_plots:
        with torch.no_grad():
            model.eval()
            # take a peek into reconstruction
            data, labels = data.to(device), labels.to(device)
            recon_batch, mu, logvar = model(data, labels)
            data = data.detach().cpu()[0]
            recon_batch= recon_batch.detach().cpu()[0]
            plt.plot(data, label='Original', color='purple')
            plt.plot(recon_batch, label='Reconstructed', color='gold')
            plt.title("Reconstruction by Decoder During Training")
            plt.xlabel("Time Step (arbitrary unit)")
            plt.ylabel("Voltage")
            plt.legend()
            plt.clf()

def test(epoch, test_losses, loader_reference, model, printout=True):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader_reference):
            data, labels = data.to(device), labels.to(device)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
    test_loss /= len(loader_reference.dataset)
    test_losses.append(test_loss)
    if printout:
        print('====> Test set loss: {:.4f}'.format(test_loss))


def val(epoch, val_losses, loader_reference, model):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader_reference):
            data, labels = data.to(device), labels.to(device)
            recon_batch, mu, logvar = model(data, labels)
            val_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
    val_loss /= len(loader_reference.dataset)
    val_losses.append(val_loss)
    print('====> Validation set loss: {:.4f}'.format(val_loss))


def initial_training():
    attempt = 0
    while attempt < 3: # Attempt more loops if initial training fails because of large loss in initial epochs
        try:
                train_losses = []
                val_losses = []
                for epoch in range(1, epochs + 1):
                        train(epoch,
                              train_losses=train_losses,
                              model=model,
                              optimizer_reference=optimizer,
                              loader_reference=train_loader,
                              make_plots=False,
                              scheduler_reference=scheduler)
                        val(epoch,
                            model=model,
                            loader_reference=val_loader,
                            val_losses=val_losses)
                plt.plot(train_losses, label = "Training Loss", color='purple')
                plt.plot(val_losses, label = "Val Loss", color='gold')
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss vs Epochs")
                plt.legend()
                plt.clf()
                train_np_arr = np.array(train_losses)
                train_np_arr = train_np_arr[train_np_arr < 100]
                plt.plot(train_losses, label = "Training Losses Less than 100", color='purple')
                plt.legend()
                plt.clf()
                return copy.deepcopy(model)
        except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}")


def generate_new_samples(num_samples, condition, model) -> np.ndarray:
    with torch.no_grad():
        model.eval()
        new_samples = []
        for i in range(num_samples):
            c = torch.tensor([condition]).to(device)
            sample = torch.randn(1, latent_size).to(device)
            sample = model.decode(sample, c).cpu().squeeze().flatten().numpy()
            new_samples.append(sample)
        return np.vstack(new_samples)


simple_rc_circuit = NaiveRCCircuit(c=1e-3, r0=1, alpha=3e3)

# Create f3dB labels for generated samples
def create_labels(new_samples: np.ndarray, circuit_system) -> np.ndarray:
    output = np.zeros(len(new_samples))
    for i, sample in enumerate(new_samples):
        arb_generator = ArbitraryWaveGenerator(1.0, 1e-3, 0.0, sample)
        f3dB = search_low_pass_3db_frequency(start_freq=1,
                                                end_freq=1e5,
                                                circuit_system=circuit_system,
                                                input_output_cycles=50,
                                                min_periods_to_equilibrium=20,
                                                waveform_generator=arb_generator)
        output[i] = f3dB

    return output


if train_initial:
    model = initial_training() # Pretrain model
    torch.save(model.state_dict(), '../models/pickled_models/most_recent_pickle.pth')


'-------------------------------- Optimization Process ----------------------------------------------'
pretrained_model = CVAE(num_voltages,
             latent_size = latent_size,
             kernel_sizes=kernels,
             use_batch_norm=True,
             channel_size=max_channel_size).to(device)


try:
    pretrained_model.load_state_dict(torch.load('../models/pickled_models/current_best_model.pth', map_location=torch.device(device)))
except Exception as e:
    print(f"Likely need to manually create current_best_model.pth by renaming a .pth pickle file. \n Exception: {e}")

pretrained_optimizer = optim.Adam(pretrained_model.parameters(), lr = learning_rate, weight_decay=0)


current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
directory_name = f'optimization_{current_date}'

base_path = './plots/optimization_plots'
full_path = os.path.join(base_path, directory_name)

os.makedirs(full_path, exist_ok=True)

# Define the subfolders
subfolders = ['reconstructions',
              'train_val_loss',
              'generated_points_per_cycle',
              'all_points',
              'f3db_diff_loss']

# Create each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(full_path, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)


make_plots = True

num_optimizations = 100
samples_per_opimization = 1000
epochs_per_optimization = 10

optimization_train_loss = []
optimization_val_loss = []

f3dB_diff_losses = []

prompt_f3db = 5
prompt_step = 5 # Amount to increase desired f3dB each optimization cycle
prompts = []
prompted_waves = []

take_x_top_percent = 50 # Choose which percentage of generated waves taken (sorted by performance)

optimization_data_set = train_data_set
# Reload the dataset from the pickle file
try:
    with open('../data/training/optimization_data_set.pkl', 'rb') as f:
        pairing = pickle.load(f)
        optimization_data_set.data, optimization_data_set.labels = pairing
except:
     pass

optimization_loader = torch.utils.data.DataLoader(dataset=optimization_data_set, batch_size=batch_size, shuffle=True)

for i in range(num_optimizations):
    print("-" * 100)
    print(f"Optimization Cycle {i} for prompt f3db={prompt_f3db} | Dataset Size {len(optimization_data_set)}")
    new_data = generate_new_samples(samples_per_opimization, prompt_f3db, pretrained_model)
    new_labels = create_labels(new_data, circuit_system=simple_rc_circuit)

    if take_x_top_percent > 0:
        # Calculate the threshold for the top 10% of the labels
        threshold = np.percentile(labels, 100 - take_x_top_percent)

        # Filter the waveforms with labels above the threshold
        top_x_percent_indices = new_labels >= threshold
        top_x_percent_waveforms = new_data[top_x_percent_indices]
        top_x_percent_labels = new_labels[top_x_percent_indices]

        new_data = top_x_percent_waveforms
        new_labels = top_x_percent_labels


    # Add new data to the dataset and shuffle
    optimization_data_set.add_data(new_data, new_labels)

      # Save the dataset to a pickle file
    with open('../data/training/optimization_data_set.pkl', 'wb') as f:
        pickle.dump(optimization_data_set._get_data(), f)

    # Reload the dataset from the pickle file
    with open('../data/training/optimization_data_set.pkl', 'rb') as f:
        pairing = pickle.load(f)
        optimization_data_set.data, optimization_data_set.labels = pairing

    # optimization_data_set = OptimizationDataset(new_data, new_labels, transform=lambda x: x.astype('float32'))

    optimization_loader = torch.utils.data.DataLoader(dataset=optimization_data_set, batch_size=batch_size, shuffle=True)

    for j in range(epochs_per_optimization):
        attempt = 0
        while attempt < 3:
                try:
                     train(j,
                           train_losses=optimization_train_loss,
                           optimizer_reference=pretrained_optimizer,
                           loader_reference=optimization_loader,
                           make_plots=make_plots,
                           model=pretrained_model,
                           scheduler_reference=None)
                     val(j,
                         val_losses=optimization_val_loss,
                         loader_reference=val_loader,
                         model=pretrained_model)
                     break
                except Exception as e:
                        attempt += 1
                        print(f"Attempt {attempt} failed with error: {e}")


    # Collect prompted waves
    with torch.no_grad():
        pretrained_model.eval()
        c = torch.tensor([prompt_f3db]).to(device)
        sample = torch.randn(1, latent_size).to(device)
        sample = pretrained_model.decode(sample, c).cpu().squeeze().flatten().numpy()
        if make_plots:
            reconstruction_path = f'./plots/optimization_plots/{directory_name}/reconstructions/f3db_prompt_{prompt_f3db}.png'
            plt.plot(sample, label="generated")
            plt.xlabel("Time")
            plt.ylabel("Voltage")
            arb_generator = ArbitraryWaveGenerator(1.0, 1e-3, 0.0, sample)
            f3dB = search_low_pass_3db_frequency(start_freq=1,
                                            end_freq=1e5,
                                            circuit_system=simple_rc_circuit,
                                            input_output_cycles=50,
                                            min_periods_to_equilibrium=20,
                                            waveform_generator=arb_generator)


            avg = np.average(new_labels)
            actual_average_f3db = round(avg, 3) # Get average of all generated labels
            f3dB_diff = np.abs(actual_average_f3db - prompt_f3db)
            f3dB_diff_losses.append(f3dB_diff)
            plt.title(f"Prompted Wave for f3dB {prompt_f3db} | Actual f3dB {round(f3dB, 3)}")
            plt.savefig(reconstruction_path)
            plt.clf()

            if i % 5 == 0:

                if i > 1:
                    f3d_diff_path = f'./plots/optimization_plots/{directory_name}/f3db_diff_loss/f3db_diff_{i}.png'
                    plt.plot(f3dB_diff_losses)
                    plt.title(f"F3dB Diff Losses over Cycles (Average of Generated) Iteration {i}")
                    plt.savefig(f3d_diff_path)
                    plt.clf()


                generated_points_path = f'./plots/optimization_plots/{directory_name}/generated_points_per_cycle/generated_points_{prompt_f3db}f3dB.png'
                plt.hist(new_labels, bins=100)
                plt.title(f"{len(new_data)} Generated Points for f3dB {prompt_f3db} with Average {round(avg, 3)}")
                plt.axvline(avg, color='r', linestyle='dashed', linewidth=2, label=f'Average: {avg:.2f}')
                plt.xlabel("f3dB")
                plt.ylabel("Count")
                plt.legend()
                plt.savefig(generated_points_path)
                plt.clf()


                all_points_path = f'./plots/optimization_plots/{directory_name}/all_points/all_points_{i}.png'
                total_f3db_avg = np.average(optimization_data_set.labels)

                plt.hist(optimization_data_set.labels, bins=100)
                plt.yscale('log')
                plt.title(f"{len(optimization_data_set.labels)} Points f3dB with Average {round(total_f3db_avg, 3)}")
                plt.axvline(total_f3db_avg, color='r', linestyle='dashed', linewidth=2, label=f'Average: {total_f3db_avg:.2f}')
                plt.xlabel("f3dB")
                plt.ylabel("Count")
                plt.legend()
                plt.savefig(all_points_path)
                plt.clf()


                if i > 1:
                    training_loss_path = f'./plots/optimization_plots/{directory_name}/train_val_loss/loss_{i}.png'
                    plt.plot(optimization_train_loss, label = "Training Loss")
                    plt.plot(optimization_val_loss, label = "Val Loss")
                    plt.xlabel("Epochs")
                    plt.ylabel("Loss")
                    plt.title(f"Training and Validation Loss vs Epochs at Optimization Cycle {i}")
                    plt.savefig(training_loss_path)
                    plt.legend()
                    plt.clf()


    prompts.append(prompt_f3db)
    prompt_f3db += prompt_step