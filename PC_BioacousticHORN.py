import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZebraFinchDataset(Dataset):
    """
    Creates sequences of MFCC frames from .wav files for next-frame prediction:
      - self.sequences[i] has shape (sequence_length, n_mfcc)
      - self.targets[i]   has the same shape, but is shifted by +1 in time
    """
    def __init__(self, root_dir, sequence_length=100, sample_rate=16000, n_mfcc=40):
        self.files = [os.path.join(root, file)
                      for root, _, files in os.walk(root_dir)
                      for file in files if file.endswith('.wav')]
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

        self.mfcc_transform = MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 40}
        )

        self.sequences = []
        self.targets = []

        print("Processing Zebra Finch Dataset...")
        for file in tqdm(self.files, desc="Loading Audio Files"):
            waveform, sr = torchaudio.load(file)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            mfcc = self.mfcc_transform(waveform).squeeze(0).transpose(0, 1)
            # Basic normalization
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-5)

            # Create sequence-target pairs with next-frame prediction
            # sequences[i] = mfcc[i : i+seq_length]
            # targets[i]   = mfcc[i+1 : i+seq_length+1]
            for i in range(len(mfcc) - self.sequence_length):
                self.sequences.append(mfcc[i : i + self.sequence_length])
                self.targets.append(mfcc[i + 1 : i + self.sequence_length + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx].float(), self.targets[idx].float()


class HarmonicOscillatorLayer(nn.Module):
    """
    A layer of size `size` that applies the dynamics:
        x_{t+1} = x_t + h*y_t
        y_{t+1} = y_t + h*(alpha * input_force - 2*gamma*y_t - omega^2*x_t)
    We do not store persistent states across the entire batch;
    the caller must handle x, y from step to step if needed.
    """
    def __init__(self, size, is_trainable=True):
        super().__init__()
        self.size = size
        self.is_trainable = is_trainable
        
        # Initialize parameters
        if self.is_trainable:
            self.omega = nn.Parameter(torch.empty(size).uniform_(1.0, 3.0))
            self.gamma = nn.Parameter(torch.empty(size).uniform_(0.1, 0.5))
            self.alpha = nn.Parameter(torch.empty(size).uniform_(0.5, 1.5))
        else:
            # For a non-trainable reservoir, store them as buffers
            self.register_buffer('omega', torch.empty(size).uniform_(1.0, 3.0))
            self.register_buffer('gamma', torch.empty(size).uniform_(0.1, 0.5))
            self.register_buffer('alpha', torch.empty(size).uniform_(0.5, 1.5))

    def forward(self, input_force, x, y, h=0.5):
        """
        input_force: (batch_size, size) or (batch_size,) if size=1
        x, y       : the current oscillator states (batch_size, size)
        h          : integration step
        Returns: (x_new, y_new)
        """
        if len(input_force.shape) == 1:
            input_force = input_force.unsqueeze(1)  # (batch_size, 1)

        # Take absolute value to ensure positivity
        omega_ = self.omega.abs()
        gamma_ = self.gamma.abs()
        alpha_ = self.alpha
        
        # Acceleration
        accel = alpha_ * input_force - 2 * gamma_ * y - (omega_ ** 2) * x

        x_new = x + h * y
        y_new = y + h * accel

        # Optional clamp to stabilize
        x_new = torch.clamp(x_new, -5.0, 5.0)
        y_new = torch.clamp(y_new, -5.0, 5.0)

        return x_new, y_new


class BioacousticHORN(nn.Module):
    """
    Architecture:
      - Input oscillator: tries to predict next input (PC)
      - Reservoir oscillator (fixed) - no PC constraints
      - Output oscillator: tries to replicate reservoir's state (PC)
      - Readout: next-frame prediction MSE
    """
    def __init__(self, input_dim=40, reservoir_size=256, seq_length=100):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.seq_length = seq_length
        
        # Input processing (trainable oscillator)
        self.input_osc = HarmonicOscillatorLayer(input_dim, is_trainable=True)
        self.input_proj = nn.Linear(input_dim, reservoir_size)
        
        # Reservoir (fixed dynamics, no PC)
        self.reservoir = HarmonicOscillatorLayer(reservoir_size, is_trainable=False)
        
        # Output processing (trainable oscillator)
        self.output_osc = HarmonicOscillatorLayer(reservoir_size, is_trainable=True)
        self.readout = nn.Linear(reservoir_size, input_dim)

    def forward(self, inputs):
        """
        inputs shape: (batch_size, seq_length, input_dim)
        We'll unroll the oscillator updates through time. 
        Returns: (predictions, pc_loss_in, pc_loss_out)
            predictions: (batch_size, seq_len, input_dim)
            pc_loss_in:  sum of MSE( x_in(t), inputs(t+1) )
            pc_loss_out: sum of MSE( x_out(t), x_res(t) )
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Initialize oscillator states to zero
        x_in = torch.zeros(batch_size, self.input_dim, device=inputs.device)
        y_in = torch.zeros(batch_size, self.input_dim, device=inputs.device)

        x_res = torch.zeros(batch_size, self.reservoir_size, device=inputs.device)
        y_res = torch.zeros(batch_size, self.reservoir_size, device=inputs.device)

        x_out = torch.zeros(batch_size, self.reservoir_size, device=inputs.device)
        y_out = torch.zeros(batch_size, self.reservoir_size, device=inputs.device)

        preds = []
        pc_loss_in = 0.0
        pc_loss_out = 0.0

        for t in range(seq_len):
            # --- Input oscillator update ---
            x_in, y_in = self.input_osc(inputs[:, t], x_in, y_in)
            
            # --- Reservoir update (fixed) ---
            proj = self.input_proj(x_in)                 # shape: (batch_size, reservoir_size)
            x_res, y_res = self.reservoir(proj, x_res, y_res)

            # --- Output oscillator update ---
            x_out, y_out = self.output_osc(x_res, x_out, y_out)

            # --- Predict next frame from output oscillator state ---
            pred = self.readout(x_out)
            preds.append(pred)

            # --- Predictive coding losses ---
            # 1) Input oscillator tries to match the NEXT input (if t < seq_len-1)
            if t < seq_len - 1:
                pc_loss_in += F.mse_loss(x_in, inputs[:, t+1])
            
            # 2) Output oscillator tries to match reservoir's current state
            pc_loss_out += F.mse_loss(x_out, x_res)

        preds = torch.stack(preds, dim=1)  # (batch_size, seq_len, input_dim)
        return preds, pc_loss_in, pc_loss_out


class BioacousticTrainer:
    def __init__(self, root_dir, batch_size=64, seq_length=100):
        self.dataset = ZebraFinchDataset(root_dir, sequence_length=seq_length)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = BioacousticHORN(
            input_dim=40, 
            reservoir_size=256, 
            seq_length=seq_length
        ).to(device)
        
        # Trainable parameters
        trainable_params = [
            {'params': self.model.input_osc.parameters()},
            {'params': self.model.output_osc.parameters()},
            {'params': self.model.input_proj.parameters()},
            {'params': self.model.readout.parameters()}
        ]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler()

        self.seq_length = seq_length

    def clamp_oscillator_params(self):
        # Keep oscillator parameters in a reasonable range
        for layer in [self.model.input_osc, self.model.output_osc]:
            layer.omega.data.clamp_(min=0.1, max=5.0)
            layer.gamma.data.clamp_(min=0.05, max=1.0)
            layer.alpha.data.clamp_(min=0.1, max=2.0)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_batches = len(self.loader)

        for seq, targets in tqdm(self.loader, desc="Training"):
            seq, targets = seq.to(device), targets.to(device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Forward pass
                preds, pc_in, pc_out = self.model(seq)  
                # MSE for next-frame prediction (preds vs. targets)
                frame_loss = F.mse_loss(preds[:, :-1], targets[:, :-1])  # or entire sequence

                # Combine losses:
                # We average the PC losses by the number of time steps to avoid huge scale
                # pc_in was summed over (seq_len-1) steps
                pc_loss_in = pc_in / (self.seq_length - 1)
                # pc_out was summed over seq_len steps
                pc_loss_out = pc_out / self.seq_length

                pc_loss = pc_loss_in + pc_loss_out
                loss = 0.7 * frame_loss + 0.3 * pc_loss

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.clamp_oscillator_params()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / total_batches

    def train(self, epochs=50):
        print("Starting training with configuration:")
        print(f"- Reservoir size: {self.model.reservoir_size}")
        print(f"- Sequence length: {self.model.seq_length}")
        print(f"- Batch size: {self.loader.batch_size}")
        print(f"- Device: {device}\n")

        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{epochs} | Total Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    trainer = BioacousticTrainer(
        root_dir="/content/drive/MyDrive/Data",
        batch_size=2048,
        seq_length=100
    )
    trainer.train(epochs=50)
