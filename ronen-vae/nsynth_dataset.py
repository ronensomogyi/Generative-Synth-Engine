import os
import json

import torch
from torch.utils.data import Dataset
import torchaudio

class NsynthDataset(Dataset):
    def __init__(self, path):
        """
        Args:
            metadata_json (str): Path to the NSynth metadata file, e.g. '/nsynth/examples.json'
            audio_dir (str): Path to the folder containing WAV files, e.g. '/nsynth/audio'
        """


        metadata_json = os.path.join(path, "examples.json")
        self.audio_dir = os.path.join(path, "audio")

        with open(metadata_json, 'r') as f:
            self.metadata = json.load(f)
    
        
        
        # Collect a list of (wav_path, label) pairs
        # We assume each key in metadata matches a wav file name in `audio_dir/`
        self.samples = []
        for example_id, attrs in self.metadata.items():
            pitch_label = attrs["pitch"]  # or any other field you'd like as a label
            if pitch_label % 12 != 0: continue # skip any non c notes
            instrument_family = attrs["instrument_family_str"]  # or any other field you'd like as a label
            wav_path = os.path.join(self.audio_dir, f"{example_id}.wav")
            self.samples.append((wav_path, instrument_family))



        # 3) Define the torchaudio MelSpectrogram transform once
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 4) Retrieve file path + label
        wav_path, instrument_family_label = self.samples[idx]

        # 5) Load the waveform from the WAV file
        #    torchaudio.load returns (waveform, sample_rate)
        waveform, sr = torchaudio.load(wav_path)

        # 6) Optionally resample if the dataset WAVs aren’t already 16 kHz
        #    Some NSynth releases are at 16 kHz, but verify your files’ sr.
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            sr = 16000

        # 7) Compute the Mel-Spectrogram then convert to log scale
        mel_spec = self.mel_transform(waveform)  # shape: [channels=1, n_mels=128, time_frames]
        log_mel_spec = torch.log(mel_spec + 1e-9)

        # 8) Return (log-mel-spectrogram, pitch_label)
        return log_mel_spec, instrument_family_label


def main():
    # Example usage
    dataset = NsynthDataset(path="/Volumes/ronen_usb/nsynth-train")
    print(f"Number of samples: {len(dataset)}")
    
    # Get a sample
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")
    import matplotlib.pyplot as plt


    # Plot 5 examples from the dataset
    for i in range(5):
        sample, label = dataset[i]
        plt.figure(figsize=(10, 4))
        plt.imshow(sample.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Example {i+1}: Instrument Family - {label}")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Channels")
        plt.colorbar(label="Log-Mel Spectrogram Amplitude")
        plt.show()

if __name__ == "__main__":
    main()