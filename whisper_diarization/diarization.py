import os
import re
import logging
import torch
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from colorama import Fore, Style
from helpers import create_config, PUNCT_MODEL_LANGS

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
        logging.warning(f"Punctuation restoration is not available for {language} language. Using the original punctuation.")
    return wsm
