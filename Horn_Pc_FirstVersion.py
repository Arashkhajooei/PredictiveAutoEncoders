import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ZebraFinchDataset(Dataset):
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
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-5)

            # Create sequence-target pairs with next-frame prediction
            for i in range(len(mfcc) - self.sequence_length):
                self.sequences.append(mfcc[i:i + self.sequence_length])
                self.targets.append(mfcc[i + 1:i + self.sequence_length + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx].float(), self.targets[idx].float()

class HarmonicOscillatorLayer(nn.Module):
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
            self.register_buffer('omega', torch.FloatTensor(size).uniform_(1.0, 3.0))
            self.register_buffer('gamma', torch.FloatTensor(size).uniform_(0.1, 0.5))
            self.register_buffer('alpha', torch.FloatTensor(size).uniform_(0.5, 1.5))

        # State initialization
        self.register_buffer('x', torch.zeros(1, size))
        self.register_buffer('y', torch.zeros(1, size))

    def forward(self, input_force):
        batch_size = input_force.size(0)
        x = self.x.repeat(batch_size, 1)
        y = self.y.repeat(batch_size, 1)
        
        # Oscillator dynamics
        accel = self.alpha * input_force - 2 * self.gamma.abs() * y - (self.omega.abs() ** 2) * x
        h = 0.5  # Fixed integration step
        x_new = x + h * y
        y_new = y + h * accel
        
        # State stabilization
        x = 0.9 * torch.clamp(x_new, -5.0, 5.0) + 0.1 * x
        y = 0.9 * torch.clamp(y_new, -5.0, 5.0) + 0.1 * y
        
        # Update persistent state
        self.x.data = x.detach().mean(dim=0, keepdim=True)
        self.y.data = y.detach().mean(dim=0, keepdim=True)
        
        return x

class BioacousticHORN(nn.Module):
    def __init__(self, input_dim=40, reservoir_size=256, seq_length=100):
        super().__init__()
        self.seq_length = seq_length
        
        # Input processing (trainable oscillators)
        self.input_osc = HarmonicOscillatorLayer(input_dim, is_trainable=True)
        self.input_proj = nn.Linear(input_dim, reservoir_size)  # NEW: Projection layer
        
        # Reservoir (fixed dynamics)
        self.reservoir = HarmonicOscillatorLayer(reservoir_size, is_trainable=False)
        
        # Output processing (trainable oscillators)
        self.output_osc = HarmonicOscillatorLayer(reservoir_size, is_trainable=True)
        self.readout = nn.Linear(reservoir_size, input_dim)

    def forward(self, inputs):
        total_pc_loss = 0.0
        predictions = []
        
        for t in range(self.seq_length):
            # Process current frame through oscillators
            input_states = self.input_osc(inputs[:, t])
            projected_input = self.input_proj(input_states)  # NEW: Project to reservoir size
            reservoir_states = self.reservoir(projected_input)
            output_states = self.output_osc(reservoir_states)
            pred = self.readout(output_states)
            predictions.append(pred)
            
            # Calculate predictive coding losses (t+1 alignment)
            if t < self.seq_length - 1:
                # Input layer target: next input frame
                input_target = inputs[:, t+1].detach()
                total_pc_loss += (input_states - input_target).pow(2).mean()
                
                # Reservoir target: projected input
                reservoir_target = projected_input.detach()  # CHANGED: Use projected input
                total_pc_loss += (reservoir_states - reservoir_target).pow(2).mean()
                
                # Output layer target: reservoir state
                output_target = reservoir_states.detach()
                total_pc_loss += (output_states - output_target).pow(2).mean()
        
        return torch.stack(predictions, dim=1), total_pc_loss

class BioacousticTrainer:
    def __init__(self, root_dir, batch_size=32, seq_length=100):
        self.dataset = ZebraFinchDataset(root_dir, sequence_length=seq_length)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = BioacousticHORN(input_dim=40, reservoir_size=256, seq_length=seq_length).to(device)
        
        # Only train oscillator parameters in input/output layers and readout
        trainable_params = [
            {'params': self.model.input_osc.parameters()},
            {'params': self.model.output_osc.parameters()},
            {'params': self.model.readout.parameters()}
        ]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for seq, targets in tqdm(self.loader, desc="Training"):
            seq, targets = seq.to(device), targets.to(device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Forward pass with next-frame prediction
                preds, pc_loss = self.model(seq)
                
                # Calculate frame prediction loss (compare predictions to next frames)
                frame_loss = nn.MSELoss()(preds[:, :-1], targets[:, 1:])
                
                # Combined loss with 7:3 ratio
                loss = 0.7 * frame_loss + 0.3 * pc_loss

            self.scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Apply parameter constraints
            with torch.no_grad():
                for layer in [self.model.input_osc, self.model.output_osc]:
                    layer.omega.data.clamp_(min=0.1, max=5.0)
                    layer.gamma.data.clamp_(min=0.05, max=1.0)
                    layer.alpha.data.clamp_(min=0.1, max=2.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(self.loader)

    def train(self, epochs=100):
        print("Starting training with configuration:")
        print(f"- Reservoir size: 256")
        print(f"- Sequence length: {self.model.seq_length}")
        print(f"- Batch size: {self.loader.batch_size}")
        print(f"- Device: {device}\n")

        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{epochs} | Total Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    trainer = BioacousticTrainer(
        root_dir="C:/Users/Snapp/PycharmProjects/Automation/bfsongrepo/extracted",
        batch_size=2048,
        seq_length=100
    )
    trainer.train(epochs=100)
