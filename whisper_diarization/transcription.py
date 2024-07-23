import logging
import torch
from colorama import Fore, Style
from whisper_diarization.ctc_forced_aligner.ctc_forced_aligner.alignment_utils import generate_emissions, get_alignments, get_spans, load_alignment_model
from whisper_diarization.ctc_forced_aligner.ctc_forced_aligner.text_utils import postprocess_results, preprocess_text
from helpers import LANGS_TO_ISO

# Update device options to exclude "mps"
device_options = {
    "cpu": "int8",
    "cuda": "float16"
}


def transcribe_audio(vocal_target, args):
    """
    Transcribes the audio file using a Whisper model, either in batches or non-batched.

    Args:
        vocal_target (str): Path to the audio file to transcribe.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        tuple: Transcription results, detected language, and audio waveform.
    """
    import whisperx

    logging.info(f"{Fore.GREEN}Initializing WhisperX model for transcription...{Style.RESET_ALL}")
    whisper_model = whisperx.load_model(
        args.model_name,
        args.device,
        compute_type=device_options[args.device],
        asr_options={"suppress_numerals": args.suppress_numerals},
    )

    logging.info(f"{Fore.GREEN}Loading audio data for transcription...{Style.RESET_ALL}")
    audio = whisperx.load_audio(vocal_target)

    logging.info(f"{Fore.GREEN}Starting transcription...{Style.RESET_ALL}")
    result = whisper_model.transcribe(audio, language=args.language, batch_size=args.batch_size)

    del whisper_model
    torch.cuda.empty_cache()

    logging.info(f"{Fore.CYAN}Transcription completed successfully!{Style.RESET_ALL}")
    return result["segments"], result["language"], audio


def perform_forced_alignment(whisper_results, language, audio_waveform, args):
    """
    Performs forced alignment of the transcribed text with the audio.

    Args:
        whisper_results (list): Transcription results from Whisper.
        language (str): Detected language.
        audio_waveform (np.array): Audio waveform data.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        list: Forced alignment results.
    """
    print(f"{Fore.BLUE}Performing forced alignment...{Style.RESET_ALL}")
    alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
        args.device, dtype=torch.float16 if args.device == "cuda" else torch.float32
    )

    audio_waveform = torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device)
    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=args.batch_size)

    del alignment_model
    torch.cuda.empty_cache()

    full_transcript = "".join(segment["text"] for segment in whisper_results)
    tokens_starred, text_starred = preprocess_text(full_transcript, romanize=True, language=LANGS_TO_ISO[language])
    segments, scores, blank_id = get_alignments(emissions, tokens_starred, alignment_dictionary)
    spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))
    return postprocess_results(text_starred, spans, stride, scores)
