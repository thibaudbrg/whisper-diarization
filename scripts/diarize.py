import argparse
import logging
import os
import re
import sys
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
    get_words_speaker_mapping, langs_to_iso, punct_model_langs,
    whisper_langs, write_srt
)
from transcription_helpers import transcribe_batched
from colorama import Fore, Style, init

init(autoreset=True)
mtypes = {"cpu": "int8", "cuda": "float16"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f"{Fore.BLUE}{Style.BRIGHT}Audio Transcription and Diarization Tool{Style.RESET_ALL}")
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
        choices=whisper_langs,
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
    if stemming:
        print(f"{Fore.MAGENTA}Isolating vocals from the rest of the audio...{Style.RESET_ALL}")
        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio}" -o "temp_outputs"')
        if return_code != 0:
            logging.warning("Source splitting failed, using original audio file. Use --no-stem argument to disable it.")
            return audio
        else:
            return os.path.join("temp_outputs", "htdemucs", os.path.splitext(os.path.basename(audio))[0], "vocals.wav")
    else:
        return audio


def transcribe_audio(vocal_target, args):
    print(f"{Fore.GREEN}Transcribing the audio file...{Style.RESET_ALL}")
    return transcribe_batched(
        vocal_target, args.language, args.batch_size,
        args.model_name, mtypes[args.device],
        args.suppress_numerals, args.device
    )


def perform_forced_alignment(whisper_results, language, audio_waveform, args):
    print(f"{Fore.BLUE}Performing forced alignment...{Style.RESET_ALL}")
    alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
        args.device, dtype=torch.float16 if args.device == "cuda" else torch.float32
    )

    audio_waveform = torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device)
    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=args.batch_size)

    del alignment_model
    torch.cuda.empty_cache()

    full_transcript = "".join(segment["text"] for segment in whisper_results)
    tokens_starred, text_starred = preprocess_text(full_transcript, romanize=True, language=langs_to_iso[language])
    segments, scores, blank_id = get_alignments(emissions, tokens_starred, alignment_dictionary)
    spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))
    return postprocess_results(text_starred, spans, stride, scores)


def save_audio_mono(audio_waveform, temp_path):
    print(f"{Fore.YELLOW}Converting audio to mono for NeMo compatibility...{Style.RESET_ALL}")
    os.makedirs(temp_path, exist_ok=True)
    audio_waveform_tensor = torch.tensor(audio_waveform).unsqueeze(0).float()  # Convert to tensor
    torchaudio.save(os.path.join(temp_path, "mono_file.wav"), audio_waveform_tensor, 16000, channels_first=True)


def perform_diarization(temp_path, args):
    print(f"{Fore.MAGENTA}Performing diarization...{Style.RESET_ALL}")
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
    msdd_model.diarize()
    del msdd_model
    torch.cuda.empty_cache()


def read_speaker_timestamps(temp_path):
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
    print(f"{Fore.GREEN}Restoring punctuation in the transcript...{Style.RESET_ALL}")
    if language in punct_model_langs:
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
            f"Punctuation restoration is not available for {language} language. Using the original punctuation.")
    return wsm


def write_output_files(ssm):
    print(f"{Fore.BLUE}Writing output files...{Style.RESET_ALL}")
    with open(f"output.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)
    with open(f"output.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)


def main():
    args = parse_arguments()
    vocal_target = isolate_vocals(args.audio, args.stemming)
    whisper_results, language, audio_waveform = transcribe_audio(vocal_target, args)
    word_timestamps = perform_forced_alignment(whisper_results, language, audio_waveform, args)
    temp_path = os.path.join(os.getcwd(), "temp_outputs")
    save_audio_mono(audio_waveform, temp_path)
    perform_diarization(temp_path, args)
    speaker_ts = read_speaker_timestamps(temp_path)
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    wsm = restore_punctuation(wsm, language)
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
    write_output_files(ssm)
    cleanup(temp_path)
    print(f"{Fore.CYAN}{Style.BRIGHT}Process completed successfully!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
