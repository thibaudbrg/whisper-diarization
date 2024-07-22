import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
import concurrent.futures

from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio

import torch
import torchaudio
from ctc_forced_aligner import (
    generate_emissions, get_alignments, get_spans,
    load_alignment_model, postprocess_results, preprocess_text
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from helpers import (
    cleanup, create_config, get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping, get_speaker_aware_transcript,
    get_words_speaker_mapping, LANGS_TO_ISO, PUNCT_MODEL_LANGS,
    WHISPER_LANGS, write_srt
)
from transcription_helpers import transcribe_batched
from colorama import Fore, Style, init

from dotenv import load_dotenv

load_dotenv()

# Initialize colorama for colored output in the terminal
init(autoreset=True)

# Update device options to exclude "mps"
device_options = {
    "cpu": "int8",
    "cuda": "float16"
}


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
        "-a", "--audio", help=f"{Fore.YELLOW}Name of the target audio file{Style.RESET_ALL}", required=True
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
            # Load the pretrained model
            model = get_model('htdemucs')
            model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.eval()

            # Load the audio file
            audio_path = Path(audio)
            wav = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
            wav = wav.mean(0).unsqueeze(0).to(model.device)  # Convert to mono and move to the correct device

            # Perform source separation
            with torch.no_grad():
                sources = model(wav)

            # Extract vocals
            vocals = sources[model.sources.index('vocals')].cpu()

            # Save the isolated vocals
            temp_output_dir = Path("temp_outputs/htdemucs")
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = temp_output_dir / f"{audio_path.stem}_vocals.wav"
            save_audio(vocals, vocals_path, model.samplerate)
            return str(vocals_path)
        except Exception as e:
            logging.warning(
                f"Source splitting failed with error: {e}. Using original audio file. Use --no-stem argument to disable it.")
            return audio
    else:
        return audio


def transcribe_audio(vocal_target, args):
    """
    Transcribes the audio file using a Whisper model.

    Args:
        vocal_target (str): Path to the audio file to transcribe.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        tuple: Transcription results, detected language, and audio waveform.
    """
    print(f"{Fore.GREEN}Transcribing the audio file...{Style.RESET_ALL}")
    return transcribe_batched(
        vocal_target, args.language, args.batch_size,
        args.model_name, device_options[args.device],
        args.suppress_numerals, args.device
    )


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


def perform_diarization(temp_path, args):
    """
    Performs speaker diarization using NeMo's MSDD model.

    Args:
        temp_path (str): Path to temporary directory for intermediate files.
        args (argparse.Namespace): Parsed command line arguments.
    """
    print(f"{Fore.MAGENTA}Performing diarization...{Style.RESET_ALL}")
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
    msdd_model.diarize()
    del msdd_model
    torch.cuda.empty_cache()


def read_speaker_timestamps(temp_path):
    """
    Reads speaker timestamps and labels from the diarization output.

    Args:
        temp_path (str): Path to temporary directory containing diarization output.

    Returns:
        list: List of speaker timestamps and labels.
    """
    print(f"{Fore.CYAN}Reading timestamps and speaker labels...{Style.RESET_ALL}")
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
    return speaker_ts


def restore_punctuation(wsm, language):
    """
    Restores punctuation in the transcript using a deep learning model.

    Args:
        wsm (list): Word-sentence mapping.
        language (str): Detected language.

    Returns:
        list: Word-sentence mapping with restored punctuation.
    """
    print(f"{Fore.GREEN}Restoring punctuation in the transcript...{Style.RESET_ALL}")
    if language in PUNCT_MODEL_LANGS:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        words_list = list(map(lambda x: x["word"], wsm))
        labeled_words = punct_model.predict(words_list, chunk_size=230)
        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if word and labeled_tuple[1] in ending_puncts and (word[-1] not in model_puncts or is_acronym(word)):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
    else:
        logging.warning(
            f"Punctuation restoration is not available for {language} language. Using the original punctuation."
        )
    return wsm


def write_output_files(ssm):
    """
    Writes the final speaker-aware transcript to text and SRT files.

    Args:
        ssm (list): Sentence-speaker mapping.
    """
    print(f"{Fore.BLUE}Writing output files...{Style.RESET_ALL}")

    # Create the outputs directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create unique filenames using the timestamp
    txt_filename = os.path.join(output_dir, f"output_{timestamp}.txt")
    srt_filename = os.path.join(output_dir, f"output_{timestamp}.srt")

    # Write the speaker-aware transcript to a text file
    with open(txt_filename, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    # Write the speaker-aware transcript to an SRT file
    with open(srt_filename, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Output files written successfully: {txt_filename}, {srt_filename}{Style.RESET_ALL}")


def main():
    """
    Main function to execute the entire audio transcription and diarization process.
    """
    args = parse_arguments()

    # Isolate vocals from the audio file if stemming is enabled
    vocal_target = isolate_vocals(args.audio, args.stemming)

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
