import os
import argparse
import torch
import numpy as np
import torchaudio.transforms as T
import soundfile as sf
from sklearn.manifold import TSNE
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import tempfile

from train_vae_gan import Encoder, Decoder, AudioDataset  # reuse components


def extract_latents(encoder, dataloader, device):
    encoder.eval()
    latents = []
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            z, _, _ = encoder(batch)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def create_tsne(latents):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(latents.numpy())
    return embedding

def synthesize_audio(decoder, z, input_dim, n_mels, max_length, n_fft, global_mean, global_std):
    decoder.eval()
    with torch.no_grad():
        fake_spec_flat = decoder(z.unsqueeze(0)).cpu().squeeze(0)

    fake_spec = fake_spec_flat.view(n_mels, max_length)
    fake_spec = fake_spec * global_std + global_mean
    fake_spec = torch.relu(fake_spec)
    fake_spec = torch.log1p(fake_spec)
    fake_spec = (fake_spec - fake_spec.min()) / (fake_spec.max() - fake_spec.min())

    fake_spec = fake_spec.unsqueeze(0).unsqueeze(0)
    fake_spec = torch.nn.functional.interpolate(fake_spec, size=(n_fft // 2 + 1, max_length), mode="bilinear")
    fake_spec = fake_spec.squeeze(0).squeeze(0)

    griffin_lim = T.GriffinLim(n_fft=n_fft, hop_length=256)
    waveform = griffin_lim(fake_spec.unsqueeze(0)).squeeze(0)

    # Save to static folder
    os.makedirs("static", exist_ok=True)
    file_name = f"sample_{np.random.randint(100000)}.wav"
    path = os.path.join("static", file_name)
    sf.write(path, waveform.numpy(), 16000)
    return path

def launch_app(tsne_coords, latents, decoder, config):
    app = Dash(__name__)
    fig = px.scatter(x=tsne_coords[:, 0], y=tsne_coords[:, 1], title="Latent Space (click to hear)")
    fig.update_traces(marker=dict(size=6), hoverinfo="skip")

    app.layout = html.Div([
        dcc.Graph(id='tsne-plot', figure=fig, style={'height': '80vh'}),
        html.Audio(id='audio-player', controls=True, src='', style={'width': '100%'})
    ])

    @app.callback(
        Output('audio-player', 'src'),
        Input('tsne-plot', 'clickData')
    )
    def update_audio(clickData):
        if clickData is None:
            return ''
        point_index = clickData['points'][0]['pointIndex']
        z = latents[point_index].to(next(decoder.parameters()).device)
        wav_path = synthesize_audio(decoder, z, **config)
        return f'/static/{os.path.basename(wav_path)}'


    # Serve audio statically
    @app.server.route('/static/<path:path>')
    def serve_static(path):
        return app.send_static_file(path)

    app.run(debug=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("audio_dir", type=str)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    # Load checkpoint and models
    checkpoint = torch.load(args.model_path)
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]
    global_mean = checkpoint["global_mean"]
    global_std = checkpoint["global_std"]

    encoder = Encoder(input_dim, latent_dim)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder = Decoder(latent_dim, input_dim)
    decoder.load_state_dict(checkpoint["decoder"])

    device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    # Dataset and latents
    dataset = AudioDataset(args.audio_dir, max_length=1000, global_mean=global_mean, global_std=global_std)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    latents = extract_latents(encoder, loader, device)[:args.max_samples]

    # Run t-SNE
    tsne_coords = create_tsne(latents)

    # Launch browser app
    launch_app(
        tsne_coords,
        latents,
        decoder,
        config=dict(
            input_dim=input_dim,
            n_mels=128,
            max_length=1000,
            n_fft=1024,
            global_mean=global_mean,
            global_std=global_std
        )
    )

if __name__ == "__main__":
    main()
