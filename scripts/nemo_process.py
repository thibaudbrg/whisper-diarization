import argparse
import os
import logging

import torch
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from pydub import AudioSegment
from colorama import Fore, Style, init

from scripts.helpers import create_config

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse arguments
parser = argparse.ArgumentParser(
    description=f"{Fore.BLUE}{Style.BRIGHT}Audio Diarization Tool{Style.RESET_ALL}"
)
parser.add_argument(
    "-a", "--audio", help=f"{Fore.YELLOW}Name of the target audio file{Style.RESET_ALL}", required=True
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help=f"{Fore.YELLOW}If you have a GPU use 'cuda', otherwise 'cpu'{Style.RESET_ALL}",
)
args = parser.parse_args()

# Convert audio to mono for NeMo compatibility
logging.info(f"{Fore.GREEN}Converting audio to mono for NeMo compatibility...{Style.RESET_ALL}")
sound = AudioSegment.from_file(args.audio).set_channels(1)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
mono_file_path = os.path.join(temp_path, "mono_file.wav")
sound.export(mono_file_path, format="wav")

# Initialize NeMo MSDD diarization model
logging.info(f"{Fore.CYAN}Initializing NeMo MSDD diarization model...{Style.RESET_ALL}")
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)

# Perform diarization
logging.info(f"{Fore.MAGENTA}Performing diarization...{Style.RESET_ALL}")
msdd_model.diarize()

logging.info(f"{Fore.CYAN}{Style.BRIGHT}Diarization process completed successfully!{Style.RESET_ALL}")
