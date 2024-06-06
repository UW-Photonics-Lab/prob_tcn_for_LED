import torch
import sys
import os
import torch.utils.data
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Add modules to path

sys.path.insert(0, "../modules/")

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

from waveforms import *

# Define sine score function to create training labels
def sin_loss(arr: np.ndarray) -> float:
    return np.sum(np.square(np.sin(((2 * np.pi) / len(arr)) * np.arange(len(arr))) - arr)) / 1e2

# # Define loss function that the decoder uses during training
# def loss_function(recon_x, x, mu, logvar):
#     MSE = F.mse_loss(recon_x, x, reduction='sum')
#     KLD = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return MSE + beta_factor * KLD

# Define custom PyTorch compatable dataset
class WaveformDataset(Dataset):
    def __init__(self, waveform_list: list[np.ndarray], label_func, transform=None):
        self.data = waveform_list
        mean = np.mean(self.data)
        std = np.std(self.data)
        self.normalize_data = (self.data - mean) / std
        self.labels = np.array([label_func(wave) for wave in self.data])
        self.transform = transform

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_labels = self.labels[idx]
        if self.transform:
            sample_data = self.transform(sample_data)
            sample_labels = self.transform(sample_labels)

        return sample_data, sample_labels
        
    def __len__(self):
        return len(self.data)

# Pytorch model definition. All Pytorch models must extend nn.Module
class CVAE(nn.Module):
    def __init__(self, 
                 feature_size, 
                 latent_size, 
                 condition_size=1, 
                 kernel_size_1=5,
                 kernel_size_2=5, 
                 kernel_size_3=5, 
                 kernel_size_4=5,  
                 padding_1 = 2,
                 padding_2 = 2,
                 channel_size = 8,# largest number of channels. Must be power of 2
                 padding_tuner = 0): #Used to tweak final output size (see final convoluation layer of decoder)
        super().__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.channel_size = channel_size


        self.encoder_convolutional_layers = nn.Sequential(
            nn.Conv1d(1, channel_size // 8, kernel_size=kernel_size_1, stride=1, padding=self._padding(kernel_size_1)),
            # nn.BatchNorm1d(channel_size // 8),
            nn.LeakyReLU(),
            nn.Conv1d(channel_size // 8, channel_size // 4, kernel_size=kernel_size_2, stride=2, padding=self._padding(kernel_size_2)),
            # nn.BatchNorm1d(channel_size // 4),
            nn.LeakyReLU(),
            nn.Conv1d(channel_size // 4, channel_size // 2, kernel_size=kernel_size_3, stride=2, padding=self._padding(kernel_size_3)),
            # nn.BatchNorm1d(channel_size // 2), 
            nn.LeakyReLU(),
            nn.Conv1d(channel_size // 2, channel_size, kernel_size=kernel_size_4, stride=2, padding=self._padding(kernel_size_4)), 
            # nn.BatchNorm1d(channel_size),
            nn.LeakyReLU(),
            nn.Flatten(), # Flatten all channels
            nn.Linear(channel_size * ((feature_size // 8) + 1), latent_size * 2) # Introduce two subarrays for mu and logvar
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_size + condition_size, channel_size * ((feature_size // 8)))
        )

        self.decoder_convolutional_layers = nn.Sequential(
            nn.ConvTranspose1d(channel_size, channel_size // 2, kernel_size=kernel_size_4, stride=2, padding=self._padding(kernel_size_4), output_padding=1),  
            # nn.BatchNorm1d(channel_size // 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(channel_size // 2, channel_size // 4, kernel_size=kernel_size_3, stride=2, padding=self._padding(kernel_size_3), output_padding=1),
            # nn.BatchNorm1d(channel_size // 4),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(channel_size // 4, channel_size // 8, kernel_size=kernel_size_2, stride=2, padding=self._padding(kernel_size_2), output_padding=1),
            # nn.BatchNorm1d(channel_size // 8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(channel_size // 8, 1, kernel_size=kernel_size_1, stride=1,  padding=(self._padding(kernel_size_1) + padding_tuner), output_padding=0),
            # nn.BatchNorm1d(1),
            nn.ReLU()
        )

        # Initialize weights with He initialization
        self._apply_he_initialization()


    def encode(self, x, c): # Q(z|x, c) function that takes x and condition, and gives us latent z
        x = torch.cat([x, c.unsqueeze(1)], 1)
        x = x.unsqueeze(1) # add single channel for convolution
        x = self.encoder_convolutional_layers(x)
        x = x.squeeze(1)
        z_mu = x[:, :self.latent_size]
        z_va = x[:, self.latent_size:]
        return z_mu, z_va

    def reparameterize(self, mu, logvar): # Reparameterization trick that samples z ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z, c): # P(x|z, c) probability distribution of x given latent z and condition c\
        z = torch.cat([z, c.unsqueeze(1)], dim=1)
        z = self.decoder_input(z)
        z = z.unsqueeze(1) # add single channel for convolution
        z = z.view(z.size(0), self.channel_size, -1)
        z = self.decoder_convolutional_layers(z)
        return z

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_predict = self.decode(z, c)
        x_predict = x_predict.squeeze(1) # remove single channel introduced for convolution
        return x_predict, mu, logvar
    
    def _apply_he_initialization(self):
        # Apply He initialization to fully connected and convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _calculate_convolution_transpose_size(self, input_size, stride, dilation, kernel_size, padding, output_padding) -> int:
        return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    
    def _calculate_convolution_size(self, input_size, padding, dilation, kernel_size, stride) -> int:
        return np.floor(((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    
    def _padding(self, kernel_size) -> int:
        return (kernel_size - 1) // 2 

    
# Defines train behavior 
def train(epoch, train_losses):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):

        data, labels = data.to(device), labels.to(device)
        recon_batch, mu, logvar = model(data, labels)

        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))


    tot_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(tot_train_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, tot_train_loss))
    
# Defines test behavior 
def test(epoch, test_losses):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))   

# Pipeline
if __name__ == "__main__":
    # Handy function that stops everything if numbers blow up or 
    # become undefined
    torch.autograd.set_detect_anomaly(True)

    if (torch.cuda.is_available()):
        device = torch.device("cuda")
    elif (torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Running on ", device)

    # Import training waveforms 
    wave_df = pd.read_csv("../data/training/prototype/non_sinusoid_waves.csv", index_col=0)
    waves = []
    for _, row in wave_df.iterrows():
        waves.append(row.to_numpy())


    # Create training labels using sine score
    labels = np.array([sin_loss(wave) for wave in waves])

    print("Average training set sine score", np.average(labels))


    # Split training data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(waves, labels, test_size=0.2, random_state=42)


    # Choose hyperparameters
    batch_size = 8
    latent_size = 5
    hidden_size = 250
    epochs = 1
    beta_factor = 0.3
    learning_rate = 1e-3

        # Build dataloaders to load in data in iterable batches
    train_data_set = WaveformDataset(X_train, sin_loss, transform=lambda x: x.astype('float32'))
    train_loader = torch.utils.data.DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)

    test_data_set = WaveformDataset(X_test, sin_loss, transform=lambda x: x.astype('float32'))
    test_loader = torch.utils.data.DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=True)

    # Create model and learning rate optimizer
    model = CVAE(100, latent_size, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    train_losses = []
    test_losses = []
    for epoch in range(1, epochs + 1):
            train(epoch, train_losses)
            test(epoch, test_losses)
    plt.plot(train_losses, label = "Training Loss")
    plt.plot(test_losses, label = "Test Loss")

    # Test the reconstruction score of 1000 prompts
    # for a perfect sine

    with torch.no_grad():
        print("Average of 1000 Sine Reconstruct Score")
        losses = []
        for i in range(1000):
            c = torch.tensor([0]).to(device) # condition for optimal sinusoid
            sample = torch.randn(1, latent_size).to(device)
            sample = model.decode(sample, c).cpu()
            loss = round(sin_loss(sample.numpy()), 2)
            losses.append(loss)
        avg_score = sum(losses) / 1000
        print(avg_score)