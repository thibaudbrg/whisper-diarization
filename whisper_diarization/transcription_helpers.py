import logging
import torch
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configure logging to display information with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe(
    audio_file: str,
    language: str,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    """
    Transcribe an audio file using the Faster Whisper model.

    Parameters:
    - audio_file (str): Path to the audio file to be transcribed.
    - language (str): Language of the audio file.
    - model_name (str): Name of the Whisper model to use.
    - compute_dtype (str): Data type for computation, e.g., 'float16', 'int8'.
    - suppress_numerals (bool): Whether to suppress numeral tokens in transcription.
    - device (str): Device to run the model on, e.g., 'cpu', 'cuda', 'mps'.

    Returns:
    - whisper_results (list): List of transcription segments.
    - info.language (str): Detected language of the audio.
    """
    from faster_whisper import WhisperModel
    from whisper_diarization.helpers import find_numeral_symbol_tokens, WAV2VEC2_LANGS

    # Initialize the Faster Whisper model
    logging.info(f"{Fore.GREEN}Initializing Faster Whisper model for non-batched transcription...{Style.RESET_ALL}")
    whisper_model = WhisperModel(model_name, device=device, compute_type=compute_dtype)

    # Optionally suppress numeral tokens
    numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer) if suppress_numerals else None

    # Determine whether to enable word timestamps based on the language
    word_timestamps = language is None or language not in WAV2VEC2_LANGS

    # Start the transcription process
    logging.info(f"{Fore.GREEN}Starting transcription...{Style.RESET_ALL}")
    segments, info = whisper_model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
        word_timestamps=word_timestamps,
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=True,
    )

    # Convert segments to dictionary format
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
    """
    Transcribe an audio file in batches using the WhisperX model.

    Parameters:
    - audio_file (str): Path to the audio file to be transcribed.
    - language (str): Language of the audio file.
    - batch_size (int): Size of batches for batched transcription.
    - model_name (str): Name of the Whisper model to use.
    - compute_dtype (str): Data type for computation, e.g., 'float16', 'int8'.
    - suppress_numerals (bool): Whether to suppress numeral tokens in transcription.
    - device (str): Device to run the model on, e.g., 'cpu', 'cuda', 'mps'.

    Returns:
    - result["segments"] (list): List of transcription segments.
    - result["language"] (str): Detected language of the audio.
    - audio: Loaded audio data.
    """
    import whisperx

    # Initialize the WhisperX model
    logging.info(f"{Fore.GREEN}Initializing WhisperX model for batched transcription...{Style.RESET_ALL}")
    whisper_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_dtype,
        asr_options={"suppress_numerals": suppress_numerals},
    )

    # Load audio data for transcription
    audio = whisperx.load_audio(audio_file)

    # Start the batched transcription process
    logging.info(f"{Fore.GREEN}Starting batched transcription...{Style.RESET_ALL}")
    result = whisper_model.transcribe(audio, language=language, batch_size=batch_size)

    # Clear GPU VRAM
    del whisper_model
    torch.cuda.empty_cache()

    logging.info(f"{Fore.CYAN}Batched transcription completed successfully!{Style.RESET_ALL}")
    return result["segments"], result["language"], audio