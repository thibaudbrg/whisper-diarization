import json
import os
import torch
import argparse
from alignment_utils import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
)
from text_utils import postprocess_results, preprocess_text

TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def parse_arguments():
    """
    Parses command line arguments for the alignment script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="CTC Forced Alignment CLI Tool")

    # Paths and language settings
    parser.add_argument("--audio_path", required=True, help="Path of the audio file")
    parser.add_argument("--text_path", required=True, help="Path of the text to be aligned")
    parser.add_argument("--language", required=True, type=str, help="Language in ISO 639-3 code")
    parser.add_argument("--romanize", action="store_true", default=False, help="Enable romanization for non-latin scripts")
    parser.add_argument("--split_size", type=str, default="word", choices=["sentence", "word", "char"], help="Alignment level")
    parser.add_argument("--star_frequency", type=str, default="edges", choices=["segment", "edges"], help="Frequency of the <star> token in the text")
    parser.add_argument("--merge_threshold", type=float, default=0.00, help="Merge segments closer than the threshold")

    # Model and computation settings
    parser.add_argument("--alignment_model", default="MahmoudAshraf/mms-300m-1130-forced-aligner", help="CTC model for alignment")
    parser.add_argument("--compute_dtype", type=str, default="float16" if torch.cuda.is_available() else "float32", choices=["bfloat16", "float16", "float32"], help="Compute dtype for alignment model inference")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--window_size", type=int, default=30, help="Window size in seconds for audio chunking")
    parser.add_argument("--context_size", type=int, default=2, help="Overlap between chunks in seconds")
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2", None], help="Attention implementation for the model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device (cuda or cpu)")

    return parser.parse_args()

def align_audio_to_text(args):
    """
    Aligns the given audio file to the provided text using a CTC model.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    model, tokenizer, dictionary = load_alignment_model(
        args.device,
        args.alignment_model,
        args.attn_implementation,
        TORCH_DTYPES[args.compute_dtype],
    )

    audio_waveform = load_audio(args.audio_path, model.dtype, model.device)
    emissions, stride = generate_emissions(model, audio_waveform, args.window_size, args.context_size, args.batch_size)

    with open(args.text_path, "r") as f:
        text = "".join(line.strip() for line in f).replace("\n", " ")

    tokens_starred, text_starred = preprocess_text(text, args.romanize, args.language, args.split_size, args.star_frequency)
    segments, scores, blank_id = get_alignments(emissions, tokens_starred, dictionary)
    spans = get_spans(tokens_starred, segments, tokenizer.decode(blank_id))
    results = postprocess_results(text_starred, spans, stride, scores, args.merge_threshold)

    output_results(args.audio_path, text, results)

def output_results(audio_path, text, results):
    """
    Outputs the alignment results to text and JSON files.

    Args:
        audio_path (str): Path of the audio file.
        text (str): The aligned text.
        results (list): The alignment results.
    """
    base_filename = os.path.splitext(audio_path)[0]
    txt_filename = f"{base_filename}.txt"
    json_filename = f"{base_filename}.json"

    with open(txt_filename, "w") as f:
        for result in results:
            f.write(f"{result['start']}-{result['end']}: {result['text']}\n")

    with open(json_filename, "w") as f:
        json.dump({"text": text, "segments": results}, f, indent=4)

def cli():
    """
    Command Line Interface for the CTC Forced Alignment Tool.
    """
    args = parse_arguments()
    align_audio_to_text(args)

if __name__ == "__main__":
    cli()
