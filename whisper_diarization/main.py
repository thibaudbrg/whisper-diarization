import argparse
import concurrent.futures
import os

import torch
from colorama import Fore, Style, init
from dotenv import load_dotenv

from audio_processing import convert_audio_to_wav, isolate_vocals, save_audio_mono
from diarization import perform_diarization, read_speaker_timestamps, restore_punctuation
from helpers import (cleanup, get_realigned_ws_mapping_with_punctuation, get_sentences_speaker_mapping,
                     write_output_files, get_words_speaker_mapping, WHISPER_LANGS)
from transcription import transcribe_audio, perform_forced_alignment

load_dotenv()
init(autoreset=True)

def parse_arguments():
    """
    Parses command line arguments provided by the user.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=f"{Fore.BLUE}{Style.BRIGHT}Audio Transcription and Diarization Tool{Style.RESET_ALL}"
    )
    parser.add_argument(
        "-a", "--audio",
        help=f"{Fore.YELLOW}Name of the target audio file{Style.RESET_ALL}",
        required=True
    )
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help=f"{Fore.GREEN}Disables source separation. This helps with long files that don't contain a lot of music.{Style.RESET_ALL}"
    )
    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help=f"{Fore.GREEN}Suppresses Numerical Digits. This helps the diarization accuracy but converts all digits into written text.{Style.RESET_ALL}"
    )
    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="medium.en",
        help=f"{Fore.YELLOW}Name of the Whisper model to use{Style.RESET_ALL}"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=8,
        help=f"{Fore.YELLOW}Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference{Style.RESET_ALL}"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=WHISPER_LANGS,
        help=f"{Fore.YELLOW}Language spoken in the audio, specify None to perform language detection{Style.RESET_ALL}"
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=f"{Fore.YELLOW}If you have a GPU use 'cuda', otherwise 'cpu'{Style.RESET_ALL}"
    )
    args = parser.parse_args()
    print(f"{Fore.CYAN}{Style.BRIGHT}Arguments parsed successfully!{Style.RESET_ALL}")
    return args

def main():
    """
    Main function to execute the entire audio transcription and diarization process.
    """
    args = parse_arguments()

    # Convert audio format to .wav and isolate vocals from the audio file if stemming is enabled
    converted_audio = convert_audio_to_wav(args.audio)
    vocal_target = isolate_vocals(converted_audio, args.stemming)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the transcription task to the executor
        whisper_future = executor.submit(transcribe_audio, vocal_target, args)

        # Retrieve results from the whisper future
        whisper_results, language, audio_waveform = whisper_future.result()

        # Save the audio in mono format for diarization
        temp_path = os.path.join(os.getcwd(), "temp_outputs")
        save_audio_mono(audio_waveform, temp_path)

        # Submit the diarization task to the executor
        diarization_future = executor.submit(perform_diarization, temp_path, args)

        # Wait for diarization to complete
        diarization_future.result()

    # Perform forced alignment of the transcribed text with the audio
    word_timestamps = perform_forced_alignment(whisper_results, language, audio_waveform, args)

    # Read diarization results and process them
    speaker_ts = read_speaker_timestamps(temp_path)
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    wsm = restore_punctuation(wsm, language)
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Write the output files
    write_output_files(ssm)
    cleanup(temp_path)

    print(f"{Fore.CYAN}{Style.BRIGHT}Process completed successfully!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
