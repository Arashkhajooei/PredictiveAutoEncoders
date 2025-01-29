import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
from tqdm import tqdm

# ============================================================
# 1) Force CUDA Usage
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

# ============================================================
# 2) Dataset Preparation
# ============================================================
class ZebraFinchDataset(Dataset):
    def __init__(self, root_dir, sequence_length=100, sample_rate=16000, n_mfcc=40):
        self.files = [os.path.join(root, file) 
                      for root, _, files in os.walk(root_dir) 
                      for file in files if file.endswith(".wav")]

        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mfcc_transform = MFCC(sample_rate=self.sample_rate, n_mfcc=n_mfcc,
                                   melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 40}).to(device)

        self.sequences = []
        self.targets = []

        print("Processing Zebra Finch Dataset...")
        for file in tqdm(self.files, desc="Loading Audio Files"):
            waveform, sr = torchaudio.load(file)
            waveform = waveform.to(device)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            mfcc = self.mfcc_transform(waveform).squeeze(0).transpose(0, 1)
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-5)

            for i in range(len(mfcc) - self.sequence_length):
                self.sequences.append(mfcc[i:i + self.sequence_length])
                self.targets.append(mfcc[i + 1:i + self.sequence_length + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx].float().to(device), self.targets[idx].float().to(device)

# ============================================================
# 3) Harmonic Oscillator Layer (Local Energy Minimization)
# ============================================================
class HarmonicOscillatorLayer(nn.Module):
    def __init__(self, size, is_trainable=True):
        super().__init__()
        self.size = size
        self.is_trainable = is_trainable

        if self.is_trainable:
            self.omega = nn.Parameter(torch.rand(size, device=device) * 2 + 1.0)
            self.gamma = nn.Parameter(torch.rand(size, device=device) * 0.4 + 0.1)
            self.alpha = nn.Parameter(torch.rand(size, device=device) * 1.0 + 0.5)
        else:
            self.register_buffer("omega", torch.rand(size, device=device) * 2 + 1.0)
            self.register_buffer("gamma", torch.rand(size, device=device) * 0.4 + 0.1)
            self.register_buffer("alpha", torch.rand(size, device=device) * 1.0 + 0.5)

    def forward(self, input_force, x, y, h=0.5):
        omega_ = self.omega.abs()
        gamma_ = self.gamma.abs()
        alpha_ = self.alpha

        accel = alpha_ * input_force - 2.0 * gamma_ * y - (omega_**2) * x
        x_new = x + h * y
        y_new = y + h * accel

        x_new = torch.clamp(x_new, -5.0, 5.0)
        y_new = torch.clamp(y_new, -5.0, 5.0)
        return x_new, y_new

# ============================================================
# 4) BioacousticHORN Model (HPC Learning)
# ============================================================
class BioacousticHORN(nn.Module):
    def __init__(self, input_dim=40, reservoir_size=256, seq_length=100):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.seq_length = seq_length

        self.input_osc = HarmonicOscillatorLayer(input_dim, is_trainable=True).to(device)
        self.input_proj = nn.Linear(input_dim, reservoir_size).to(device)

        self.reservoir = HarmonicOscillatorLayer(reservoir_size, is_trainable=False).to(device)
        self.output_osc = HarmonicOscillatorLayer(reservoir_size, is_trainable=True).to(device)
        self.readout = nn.Linear(reservoir_size, input_dim).to(device)

    def forward_unroll(self, inputs):
        B, T, _ = inputs.shape

        x_in = torch.zeros(B, self.input_dim, device=device)
        y_in = torch.zeros(B, self.input_dim, device=device)
        x_res = torch.zeros(B, self.reservoir_size, device=device)
        y_res = torch.zeros(B, self.reservoir_size, device=device)
        x_out = torch.zeros(B, self.reservoir_size, device=device)
        y_out = torch.zeros(B, self.reservoir_size, device=device)

        x_in_all, x_res_all, x_out_all, preds = [], [], [], []
        
        for t in range(T):
            x_in, y_in = self.input_osc(inputs[:, t], x_in, y_in)
            proj = self.input_proj(x_in)
            x_res, y_res = self.reservoir(proj, x_res, y_res)
            x_out, y_out = self.output_osc(x_res, x_out, y_out)
            preds.append(self.readout(x_out))

            x_in_all.append(x_in)
            x_res_all.append(x_res)
            x_out_all.append(x_out)

        return torch.stack(preds, dim=1), torch.stack(x_in_all, dim=1), torch.stack(x_res_all, dim=1), torch.stack(x_out_all, dim=1)

    def compute_energy(self, inputs, preds, x_in, x_res, x_out):
        if inputs.shape[1] > 1:
            pc_loss_in = ((x_in[:, :-1] - inputs[:, 1:])**2).mean()
            frame_loss = ((preds[:, :-1] - inputs[:, 1:])**2).mean()
        else:
            pc_loss_in = frame_loss = torch.tensor(0.0, device=device)

        pc_loss_out = ((x_out - x_res)**2).mean()
        return frame_loss + pc_loss_in + pc_loss_out

    def local_update_params(self, energy, lr=1e-3):
        for layer in [self.input_osc, self.output_osc]:
            for param in [layer.omega, layer.gamma, layer.alpha]:
                if param.requires_grad:
                    param.data -= lr * (energy.detach().item() * torch.sign(param.data) - 0.01 * param.data)
                    param.data.clamp_(-5.0, 5.0)

# ============================================================
# 5) Trainer (HPC Training with Improved Weight Update)
# ============================================================
class BioacousticTrainer:
    def __init__(self, root_dir, batch_size=64, seq_length=100):
        self.dataset = ZebraFinchDataset(root_dir, sequence_length=seq_length)

        # ✅ Explicitly set the generator to CUDA
        generator = torch.Generator(device=device)
        
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, generator=generator)

        self.model = BioacousticHORN(input_dim=40, reservoir_size=256, seq_length=seq_length).to(device)


    def train_epoch(self, lr=1e-3):
        self.model.train()
        total_energy = 0.0

        for seq, _ in tqdm(self.loader, desc="Training HPC"):
            seq = seq.to(device)
            preds, x_in_t, x_res_t, x_out_t = self.model.forward_unroll(seq)
            energy = self.model.compute_energy(seq, preds, x_in_t, x_res_t, x_out_t)
            self.model.local_update_params(energy, lr)
            total_energy += energy.item()

        return total_energy / len(self.loader)

    def train(self, epochs=10, lr=1e-3):
        for epoch in range(epochs):
            avg_energy = self.train_epoch(lr)
            print(f"Epoch {epoch+1}/{epochs} | Avg Energy: {avg_energy:.4f}")

# ============================================================
# 6) Run Training
# ============================================================
if __name__ == "__main__":
    trainer = BioacousticTrainer(root_dir=r"C:\Users\Snapp\PycharmProjects\Automation\bfsongrepo\extracted", batch_size=2048, seq_length=100)
    trainer.train(epochs=10, lr=1e-3)
