import logging

import torch
from colorama import Fore, Style, init
import logging

# Initialize colorama
init(autoreset=True)

def transcribe(
    audio_file: str,
    language: str,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    from faster_whisper import WhisperModel
    from scripts.helpers import find_numeral_symbol_tokens, WAV2VEC2_LANGS

    logging.info(f"{Fore.GREEN}Initializing Faster Whisper model for non-batched transcription...{Style.RESET_ALL}")
    whisper_model = WhisperModel(model_name, device=device, compute_type=compute_dtype)

    if suppress_numerals:
        numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    else:
        numeral_symbol_tokens = None

    word_timestamps = language is None or language not in WAV2VEC2_LANGS

    logging.info(f"{Fore.GREEN}Starting transcription...{Style.RESET_ALL}")
    segments, info = whisper_model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
        word_timestamps=word_timestamps,
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=True,
    )

    whisper_results = [segment._asdict() for segment in segments]

    # Clear GPU VRAM
    del whisper_model
    torch.cuda.empty_cache()

    logging.info(f"{Fore.CYAN}Transcription completed successfully!{Style.RESET_ALL}")
    return whisper_results, info.language

def transcribe_batched(
    audio_file: str,
    language: str,
    batch_size: int,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    import whisperx

    logging.info(f"{Fore.GREEN}Initializing WhisperX model for batched transcription...{Style.RESET_ALL}")
    whisper_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_dtype,
        asr_options={"suppress_numerals": suppress_numerals},
    )

    audio = whisperx.load_audio(audio_file)

    logging.info(f"{Fore.GREEN}Starting batched transcription...{Style.RESET_ALL}")
    result = whisper_model.transcribe(audio, language=language, batch_size=batch_size)

    # Clear GPU VRAM
    del whisper_model
    torch.cuda.empty_cache()

    logging.info(f"{Fore.CYAN}Batched transcription completed successfully!{Style.RESET_ALL}")
    return result["segments"], result["language"], audio
