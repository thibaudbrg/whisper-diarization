import json
import logging
import os
import shutil
from datetime import datetime

import nltk
import wget
from colorama import Fore, Style, init
from omegaconf import OmegaConf
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PUNCT_MODEL_LANGS = ["en", "fr", "de", "es", "it", "nl", "pt", "bg", "pl", "cs", "sk", "sl"]
WAV2VEC2_LANGS = list(DEFAULT_ALIGN_MODELS_TORCH.keys()) + list(DEFAULT_ALIGN_MODELS_HF.keys())
WHISPER_LANGS = sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()])
LANGS_TO_ISO = {
    "aa": "aar", "ab": "abk", "ae": "ave", "af": "afr", "ak": "aka", "am": "amh", "an": "arg",
    "ar": "ara", "as": "asm", "av": "ava", "ay": "aym", "az": "aze", "ba": "bak", "be": "bel",
    "bg": "bul", "bh": "bih", "bi": "bis", "bm": "bam", "bn": "ben", "bo": "tib", "br": "bre",
    "bs": "bos", "ca": "cat", "ce": "che", "ch": "cha", "co": "cos", "cr": "cre", "cs": "cze",
    "cu": "chu", "cv": "chv", "cy": "wel", "da": "dan", "de": "ger", "dv": "div", "dz": "dzo",
    "ee": "ewe", "el": "gre", "en": "eng", "eo": "epo", "es": "spa", "et": "est", "eu": "baq",
    "fa": "per", "ff": "ful", "fi": "fin", "fj": "fij", "fo": "fao", "fr": "fre", "fy": "fry",
    "ga": "gle", "gd": "gla", "gl": "glg", "gn": "grn", "gu": "guj", "gv": "glv", "ha": "hau",
    "he": "heb", "hi": "hin", "ho": "hmo", "hr": "hrv", "ht": "hat", "hu": "hun", "hy": "arm",
    "hz": "her", "ia": "ina", "id": "ind", "ie": "ile", "ig": "ibo", "ii": "iii", "ik": "ipk",
    "io": "ido", "is": "ice", "it": "ita", "iu": "iku", "ja": "jpn", "jv": "jav", "ka": "geo",
    "kg": "kon", "ki": "kik", "kj": "kua", "kk": "kaz", "kl": "kal", "km": "khm", "kn": "kan",
    "ko": "kor", "kr": "kau", "ks": "kas", "ku": "kur", "kv": "kom", "kw": "cor", "ky": "kir",
    "la": "lat", "lb": "ltz", "lg": "lug", "li": "lim", "ln": "lin", "lo": "lao", "lt": "lit",
    "lu": "lub", "lv": "lav", "mg": "mlg", "mh": "mah", "mi": "mao", "mk": "mac", "ml": "mal",
    "mn": "mon", "mr": "mar", "ms": "may", "mt": "mlt", "my": "bur", "na": "nau", "nb": "nob",
    "nd": "nde", "ne": "nep", "ng": "ndo", "nl": "dut", "nn": "nno", "no": "nor", "nr": "nbl",
    "nv": "nav", "ny": "nya", "oc": "oci", "oj": "oji", "om": "orm", "or": "ori", "os": "oss",
    "pa": "pan", "pi": "pli", "pl": "pol", "ps": "pus", "pt": "por", "qu": "que", "rm": "roh",
    "rn": "run", "ro": "rum", "ru": "rus", "rw": "kin", "sa": "san", "sc": "srd", "sd": "snd",
    "se": "sme", "sg": "sag", "si": "sin", "sk": "slo", "sl": "slv", "sm": "smo", "sn": "sna",
    "so": "som", "sq": "alb", "sr": "srp", "ss": "ssw", "st": "sot", "su": "sun", "sv": "swe",
    "sw": "swa", "ta": "tam", "te": "tel", "tg": "tgk", "th": "tha", "ti": "tir", "tk": "tuk",
    "tl": "tgl", "tn": "tsn", "to": "ton", "tr": "tur", "ts": "tso", "tt": "tat", "tw": "twi",
    "ty": "tah", "ug": "uig", "uk": "ukr", "ur": "urd", "uz": "uzb", "ve": "ven", "vi": "vie",
    "vo": "vol", "wa": "wln", "wo": "wol", "xh": "xho", "yi": "yid", "yo": "yor", "za": "zha",
    "zh": "chi", "zu": "zul",
}

def create_config(output_dir):
    """
    Creates the configuration required for NeMo Diarization, including downloading the configuration file if it does not exist locally.

    Args:
        output_dir (str): Directory where intermediate files and prediction outputs will be stored.

    Returns:
        OmegaConf: Loaded configuration object for NeMo diarization.
    """
    logging.info(f"{Fore.CYAN}Creating configuration for NeMo Diarization{Style.RESET_ALL}")
    DOMAIN_TYPE = "telephonic"
    CONFIG_LOCAL_DIRECTORY = "config"
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    MODEL_CONFIG_PATH = os.path.join(CONFIG_LOCAL_DIRECTORY, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG_PATH):
        os.makedirs(CONFIG_LOCAL_DIRECTORY, exist_ok=True)
        CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
        MODEL_CONFIG_PATH = wget.download(CONFIG_URL, MODEL_CONFIG_PATH)
    config = OmegaConf.load(MODEL_CONFIG_PATH)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")
    config.num_workers = 0
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = output_dir
    config.diarizer.speaker_embeddings.model_path = "titanet_large"
    config.diarizer.oracle_vad = False
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    config.diarizer.vad.model_path = "vad_multilingual_marblenet"
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"
    return config

def get_word_ts_anchor(s, e, option="start"):
    """
    Get the anchor timestamp for a word based on the specified option.

    Args:
        s (int): Start time of the word in milliseconds.
        e (int): End time of the word in milliseconds.
        option (str): Option to choose the anchor point. Can be 'start', 'mid', or 'end'.

    Returns:
        int: The anchor timestamp for the word.
    """
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s

def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    """
    Maps words to their respective speakers based on timestamps.

    Args:
        wrd_ts (list): List of word timestamps.
        spk_ts (list): List of speaker timestamps.
        word_anchor_option (str): Option to choose the anchor point for words. Can be 'start', 'mid', or 'end'.

    Returns:
        list: List of dictionaries with word, start_time, end_time, and speaker.
    """
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = int(wrd_dict["start"] * 1000), int(wrd_dict["end"] * 1000), wrd_dict["text"]
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append({"word": wrd, "start_time": ws, "end_time": we, "speaker": sp})
    return wrd_spk_mapping

def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    """
    Get the index of the first word in a sentence based on the maximum number of words and speaker continuity.

    Args:
        word_idx (int): Index of the current word.
        word_list (list): List of words.
        speaker_list (list): List of speakers corresponding to the words.
        max_words (int): Maximum number of words in a sentence.

    Returns:
        int: Index of the first word in the sentence.
    """
    is_word_sentence_end = lambda x: x >= 0 and word_list[x][-1] in ".?!"
    left_idx = word_idx
    while left_idx > 0 and word_idx - left_idx < max_words and speaker_list[left_idx - 1] == speaker_list[left_idx] and not is_word_sentence_end(left_idx - 1):
        left_idx -= 1
    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1

def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    """
    Get the index of the last word in a sentence based on the maximum number of words.

    Args:
        word_idx (int): Index of the current word.
        word_list (list): List of words.
        max_words (int): Maximum number of words in a sentence.

    Returns:
        int: Index of the last word in the sentence.
    """
    is_word_sentence_end = lambda x: x >= 0 and word_list[x][-1] in ".?!"
    right_idx = word_idx
    while right_idx < len(word_list) - 1 and right_idx - word_idx < max_words and not is_word_sentence_end(right_idx):
        right_idx += 1
    return right_idx if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx) else -1

def get_realigned_ws_mapping_with_punctuation(word_speaker_mapping, max_words_in_sentence=50):
    """
    Realign the word-speaker mapping with punctuation, ensuring sentence continuity and speaker consistency.

    Args:
        word_speaker_mapping (list): List of dictionaries with word, start_time, end_time, and speaker.
        max_words_in_sentence (int): Maximum number of words in a sentence.

    Returns:
        list: Realigned list of dictionaries with word, start_time, end_time, and speaker.
    """
    is_word_sentence_end = lambda x: x >= 0 and word_speaker_mapping[x]["word"][-1] in ".?!"
    wsp_len = len(word_speaker_mapping)
    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)
    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if k < wsp_len - 1 and speaker_list[k] != speaker_list[k + 1] and not is_word_sentence_end(k):
            left_idx = get_first_word_idx_of_sentence(k, words_list, speaker_list, max_words_in_sentence)
            right_idx = get_last_word_idx_of_sentence(k, words_list, max_words_in_sentence - k + left_idx - 1) if left_idx > -1 else -1
            if min(left_idx, right_idx) == -1:
                k += 1
                continue
            spk_labels = speaker_list[left_idx:right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue
            speaker_list[left_idx:right_idx + 1] = [mod_speaker] * (right_idx - left_idx + 1)
            k = right_idx
        k += 1
    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1
    return realigned_list

def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    """
    Get the mapping of sentences to their respective speakers.

    Args:
        word_speaker_mapping (list): List of dictionaries with word, start_time, end_time, and speaker.
        spk_ts (list): List of speaker timestamps.

    Returns:
        list: List of dictionaries with speaker, start_time, end_time, and text.
    """
    sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
    s, e, spk = spk_ts[0]
    prev_spk = spk
    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}
    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk or sentence_checker(snt["text"] + " " + wrd):
            snts.append(snt)
            snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk
    snts.append(snt)
    return snts

def get_speaker_aware_transcript(sentences_speaker_mapping, f):
    """
    Writes the speaker-aware transcript to a file.

    Args:
        sentences_speaker_mapping (list): List of dictionaries with speaker, start_time, end_time, and text.
        f (file object): File object to write the transcript to.
    """
    previous_speaker = sentences_speaker_mapping[0]["speaker"]
    f.write(f"{previous_speaker}: ")
    for sentence_dict in sentences_speaker_mapping:
        speaker = sentence_dict["speaker"]
        sentence = sentence_dict["text"]
        if speaker != previous_speaker:
            f.write(f"\n\n{speaker}: ")
            previous_speaker = speaker
        f.write(sentence + " ")

def format_timestamp(milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    """
    Formats a timestamp from milliseconds to a string in the format HH:MM:SS,mmm.

    Args:
        milliseconds (float): Time in milliseconds.
        always_include_hours (bool): Whether to always include the hours part in the timestamp.
        decimal_marker (str): Character to use as the decimal marker.

    Returns:
        str: Formatted timestamp string.
    """
    assert milliseconds >= 0, "non-negative timestamp expected"
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

def write_srt(transcript, file):
    """
    Writes a transcript to a file in SRT (SubRip Subtitle) format.

    Args:
        transcript (list): List of dictionaries with start_time, end_time, speaker, and text.
        file (file object): File object to write the SRT content to.
    """
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

def find_numeral_symbol_tokens(tokenizer):
    """
    Finds tokens in the tokenizer vocabulary that contain numeral or symbol characters.

    Args:
        tokenizer (Tokenizer): Tokenizer object.

    Returns:
        list: List of token IDs that contain numeral or symbol characters.
    """
    return [token_id for token, token_id in tokenizer.get_vocab().items() if any(c in "0123456789%$£" for c in token)]

def _get_next_start_timestamp(word_timestamps, current_word_index, final_timestamp):
    """
    Gets the start timestamp of the next word.

    Args:
        word_timestamps (list): List of word timestamps.
        current_word_index (int): Index of the current word.
        final_timestamp (float): Final timestamp of the audio.

    Returns:
        float: Start timestamp of the next word.
    """
    if current_word_index == len(word_timestamps) - 1:
        return word_timestamps[current_word_index]["start"]
    next_word_index = current_word_index + 1
    while current_word_index < len(word_timestamps) - 1:
        if word_timestamps[next_word_index].get("start") is None:
            word_timestamps[current_word_index]["word"] += " " + word_timestamps[next_word_index]["word"]
            word_timestamps[next_word_index]["word"] = None
            next_word_index += 1
            if next_word_index == len(word_timestamps):
                return final_timestamp
        else:
            return word_timestamps[next_word_index]["start"]

def filter_missing_timestamps(word_timestamps, initial_timestamp=0, final_timestamp=None):
    """
    Filters out missing timestamps and fills them in based on adjacent timestamps.

    Args:
        word_timestamps (list): List of word timestamps.
        initial_timestamp (float): Initial timestamp to use if the first word has no timestamp.
        final_timestamp (float): Final timestamp to use if the last word has no timestamp.

    Returns:
        list: List of word timestamps with missing timestamps filled in.
    """
    if word_timestamps[0].get("start") is None:
        word_timestamps[0]["start"] = initial_timestamp if initial_timestamp is not None else 0
        word_timestamps[0]["end"] = _get_next_start_timestamp(word_timestamps, 0, final_timestamp)
    result = [word_timestamps[0]]
    for i, ws in enumerate(word_timestamps[1:], start=1):
        if ws.get("start") is None and ws.get("word") is not None:
            ws["start"] = word_timestamps[i - 1]["end"]
            ws["end"] = _get_next_start_timestamp(word_timestamps, i, final_timestamp)
        if ws["word"] is not None:
            result.append(ws)
    return result

def cleanup(path: str):
    """
    Cleans up the specified path by removing the file or directory.

    Args:
        path (str): Path to the file or directory to be removed.

    Raises:
        ValueError: If the specified path is not a file or directory.
    """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError(f"Path {path} is not a file or dir.")

def process_language_arg(language: str, model_name: str):
    """
    Processes the language argument to ensure it is valid and converts language names to language codes.

    Args:
        language (str): Language specified by the user.
        model_name (str): Name of the model being used.

    Returns:
        str: Processed language code.

    Raises:
        ValueError: If the language is not supported.
    """
    if language is not None:
        language = language.lower()
    if language not in LANGUAGES:
        if language in TO_LANGUAGE_CODE:
            language = TO_LANGUAGE_CODE[language]
        else:
            raise ValueError(f"Unsupported language: {language}")
    if model_name.endswith(".en") and language != "en":
        if language is not None:
            logging.warning(f"{model_name} is an English-only model but received '{language}'; using English instead.")
        language = "en"
    return language

def write_output_files(ssm, whisper_model):
    """
    Writes the final speaker-aware transcript to text and SRT files, including the Whisper model used.

    Args:
        ssm (list): Sentence-speaker mapping.
        whisper_model (str): The name of the Whisper model used.
    """
    print(f"{Fore.BLUE}Writing output files...{Style.RESET_ALL}")

    # Create the outputs directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create unique filenames using the timestamp and Whisper model name
    txt_filename = os.path.join(output_dir, f"output_{timestamp}_{whisper_model}.txt")
    srt_filename = os.path.join(output_dir, f"output_{timestamp}_{whisper_model}.srt")

    # Write the speaker-aware transcript to a text file
    with open(txt_filename, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    # Write the speaker-aware transcript to an SRT file
    with open(srt_filename, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    print(f"{Fore.CYAN}{Style.BRIGHT}Output files written successfully: {txt_filename}, {srt_filename}{Style.RESET_ALL}")

    # Return the paths to the output files
    return txt_filename, srt_filename