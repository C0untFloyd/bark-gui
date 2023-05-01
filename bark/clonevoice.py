from bark.generation import load_codec_model, generate_text_semantic, grab_best_device
from encodec.utils import convert_audio
import torchaudio
import torch
import os
import gradio


def clone_voice(audio_filepath, text, dest_filename, progress=gradio.Progress(track_tqdm=True)):
    if len(text) < 1:
        raise gradio.Error('No transcription text entered!')

    use_gpu = not os.environ.get("BARK_FORCE_CPU", False)
    progress(0, desc="Loading Codec")
    model = load_codec_model(use_gpu=use_gpu)
    progress(0.25, desc="Converting WAV")

    # Load and pre-process the audio waveform
    device = grab_best_device(use_gpu)
    wav, sr = torchaudio.load(audio_filepath)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)
    progress(0.5, desc="Extracting codes")

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # get seconds of audio
    seconds = wav.shape[-1] / model.sample_rate
    # generate semantic tokens
    semantic_tokens = generate_text_semantic(text, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7)

    # move codes to cpu
    codes = codes.cpu().numpy()

    import numpy as np
    output_path = dest_filename + '.npz'
    np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
    return "Finished"
