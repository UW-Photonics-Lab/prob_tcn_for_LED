import torch
import os
import zarr
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Optional

@dataclass
class OFDM_channel:
    FREQUENCIES: Optional[torch.Tensor] = None
    NUM_POINTS_SYMBOL: Optional[int] = None
    CP_LENGTH: Optional[int] = None
    DELTA_K: Optional[int] = None
    KS: Optional[torch.Tensor] = None
    K_MIN: Optional[int] = None
    K_MAX: Optional[int] = None
    LEFT_PADDING_ZEROS: Optional[int] = None
    CP_RATIO: Optional[float] = None
    NUM_POINTS_FRAME: Optional[int] = None
    NUM_POS_FREQS: Optional[int] = None
    RIGHT_PADDING_ZEROS: Optional[int] = None
    sent_frames_freq: Optional[torch.Tensor] = None
    received_frames_freq: Optional[torch.Tensor] = None
    sent_frames_time: Optional[torch.Tensor] = None
    received_frames_time: Optional[torch.Tensor] = None

def symbols_to_time(X,
                    num_right_padding_zeros: int,
                    num_left_padding_zeros=0,
                    negative_rail=-3.0,
                    positive_rail=3.0):
        'Convert OFDM symbols to real valued signal'
        # Make hermetian symmetric
        Nt, Nf = X.shape
        device = X.device
        num_right_padding_zeros = torch.zeros(Nt, num_right_padding_zeros)
        num_left_padding_zeros = torch.zeros(Nt, num_left_padding_zeros)
        X = torch.cat([num_left_padding_zeros, X, num_right_padding_zeros], dim=-1)
        DC_Nyquist = torch.zeros((X.shape[0], 1))
        X_hermitian = torch.flip(X, dims=[1]).conj()
        X_full = torch.hstack([DC_Nyquist, X, DC_Nyquist, X_hermitian])

        # Convert to time domain
        x_time = torch.fft.ifft(X_full, dim=-1, norm="ortho").real
        x_time = torch.clip(x_time, min=negative_rail, max=positive_rail)
        return x_time.to(device)

def extract_zarr_data(file_path, device, delay=None):
    o_sets = OFDM_channel()
    cache_path = file_path.replace(".zarr", "_cached.pt").replace(".h5", "_cached.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location=device)
        o_sets.sent_frames_time = data["sent_frames_time"].to(device)
        o_sets.received_frames_time = data["received_frames_time"].to(device)
        o_sets.FREQUENCIES = data["frequencies"].to(device)
        o_sets.NUM_POINTS_SYMBOL = data["NUM_POINTS_SYMBOL"]
        o_sets.CP_LENGTH = data["CP_LENGTH"]
        print("Loaded from cache!")

    else:
        print("No cache found — loading original dataset...")
        root = zarr.open(file_path, mode="r")
        sent, received, received_time = [], [], []

        # Loop through frames
        num_skipped = 0
        for frame_key in root.group_keys():
            try:
                frame = root[frame_key]
                if o_sets.FREQUENCIES is None:
                    o_sets.FREQUENCIES = torch.tensor(frame["freqs"][:], dtype=torch.int).real
                    o_sets.NUM_POINTS_SYMBOL = int(frame.attrs["num_points_symbol"])
                    o_sets.CP_LENGTH = int(frame.attrs["cp_length"])
                else:
                    pass

                sent.append(torch.tensor(frame["sent"][:], dtype=torch.complex64))
                received.append(torch.tensor(frame["received"][:], dtype=torch.complex64))
                if "received_time" in frame:
                    received_time.append(torch.tensor(frame["received_time"][:], dtype=torch.float32))
            except:
                num_skipped += 1
                pass # skip corrupted frames
        print(f"Skipped {num_skipped} corrupted frames")

        o_sets.sent_frames_freq = torch.stack(sent).squeeze(1)
        o_sets.received_frames_freq = torch.stack(received).squeeze(1)
        o_sets.DELTA_K = o_sets.FREQUENCIES[1] - o_sets.FREQUENCIES[0]
        o_sets.KS = (o_sets.FREQUENCIES / o_sets.DELTA_K).to(torch.int)
        o_sets.K_MIN = o_sets.KS[0]
        o_sets.K_MAX = o_sets.KS[-1]
        o_sets.LEFT_PADDING_ZEROS = o_sets.K_MIN - 1 # don't include DC freq
        o_sets.NUM_POS_FREQS = o_sets.K_MAX + 1
        o_sets.NUM_POINTS_FRAME = o_sets.NUM_POINTS_SYMBOL + o_sets.CP_LENGTH
        o_sets.RIGHT_PADDING_ZEROS = (o_sets.NUM_POINTS_FRAME  - 2 * o_sets.NUM_POS_FREQS) // 2

        # Handle received time symbols; perform some cleaning if necessary
        N_shortest = min(t.size(-1) for t in received_time)
        good_indices = [i for i, x in enumerate(received_time) if x.size(-1) == N_shortest]
        received_frames_time = torch.stack([t for t in received_time if t.size(-1) == N_shortest],dim=0).real
        sent_frames_freq = o_sets.sent_frames_freq[good_indices]
        o_sets.received_frames_time = received_frames_time.squeeze(1)
        o_sets.sent_frames_time = symbols_to_time(sent_frames_freq,
                                           o_sets.LEFT_PADDING_ZEROS,
                                           o_sets.RIGHT_PADDING_ZEROS)

        cache_path = file_path.replace(".zarr", "_cached.pt").replace(".h5", "_cached.pt")
        torch.save({
            "sent_frames_time": o_sets.sent_frames_time.cpu(),
            "received_frames_time": received_frames_time.cpu(),
            "frequencies": o_sets.FREQUENCIES.cpu(),
            "NUM_POINTS_SYMBOL": o_sets.NUM_POINTS_SYMBOL,
            "CP_LENGTH": o_sets.CP_LENGTH
        }, cache_path)

    return o_sets

class ChannelData(Dataset):
    def __init__(self,
                sent_frames,
                received_frames,
                frequencies,
                transform=None,
                target_transform=None):

        self.sent_frames = sent_frames
        self.received_frames = received_frames
        assert len(sent_frames) == len(received_frames)

    def __len__(self):
        return len(self.sent_frames)

    def __getitem__(self, idx):
        return self.sent_frames[idx], self.received_frames[idx]
