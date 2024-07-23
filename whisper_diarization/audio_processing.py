import logging
import os
from pathlib import Path

import torchaudio
from pydub import AudioSegment
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio
import torch
from colorama import Fore, Style

def convert_audio_to_wav(input_audio):
    """
    Converts any audio format to WAV format with the desired specifications.

    Args:
        input_audio (str): Path to the input audio file.

    Returns:
        str: Path to the converted WAV audio file.
    """
    output_dir = "temp_outputs"
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_file(input_audio)
    audio = audio.set_channels(1).set_frame_rate(16000)
    output_path = os.path.join(output_dir, "converted_audio.wav")
    audio.export(output_path, format="wav")
    print(f"{Fore.CYAN}Audio file converted to .wav...{Style.RESET_ALL}")
    return output_path

def isolate_vocals(audio, stemming):
    """
    Isolates the vocal track from the rest of the audio using Demucs, a music source separation tool.

    Args:
        audio (str): Path to the audio file.
        stemming (bool): Whether to perform source separation.

    Returns:
        str: Path to the isolated vocals or original audio if source separation fails.
    """
    if stemming:
        print(f"{Fore.MAGENTA}Isolating vocals from the rest of the audio...{Style.RESET_ALL}")
        try:
            model = get_model('htdemucs')
            model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.eval()
            audio_path = Path(audio)
            wav = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
            wav = wav.mean(0).unsqueeze(0).to(model.device)
            with torch.no_grad():
                sources = model(wav)
            vocals = sources[model.sources.index('vocals')].cpu()
            temp_output_dir = Path("temp_outputs/htdemucs")
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = temp_output_dir / f"{audio_path.stem}_vocals.wav"
            save_audio(vocals, vocals_path, model.samplerate)
            return str(vocals_path)
        except Exception as e:
            logging.warning(
                f"Source splitting failed with error: {e}. Using original audio file. Use --no-stem argument to "
                f"disable it.")
            return audio
    else:
        return audio

def save_audio_mono(audio_waveform, temp_path):
    """
    Converts the audio waveform to mono for NeMo compatibility and saves it to disk.

    Args:
        audio_waveform (np.array): Audio waveform data.
        temp_path (str): Path to save the mono audio file.
    """
    print(f"{Fore.YELLOW}Converting audio to mono for NeMo compatibility...{Style.RESET_ALL}")
    os.makedirs(temp_path, exist_ok=True)
    audio_waveform_tensor = torch.tensor(audio_waveform).unsqueeze(0).float()
    torchaudio.save(os.path.join(temp_path, "mono_file.wav"), audio_waveform_tensor, 16000, channels_first=True)
