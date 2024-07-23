# Whisper + Diarization using NeMo 

## Project Overview 

This project combines advanced speech recognition and speaker diarization techniques to transcribe and identify speakers in audio recordings. We use OpenAI's Whisper model for transcription and NVIDIA's NeMo MSDD model for speaker diarization. The project can process various types of audio, including telephonic, meeting, and general conversations, with high accuracy and efficiency.

## Installation 

### Prerequisites 

- Python 3.10
- CUDA-enabled GPU (optional but recommended for faster processing)

### Setup 
 
1. Clone the repository:

```bash
git clone https://github.com/thibaudbrg/whisper-diarization.git
cd whisper-diarization
```

1. Install dependencies using Poetry:


```bash
poetry install
```
 
1. Configure your environment variables by creating a `.env` file (if necessary):


```bash
cp .env.example .env
```
 
1. Build the `ctc_forced_aligner` C++ library:


```bash
python whisper_diarization/ctc_forced_aligner/setup.py build_ext --inplace
```

## Usage 

### Command Line Interface 
Run the main script `concurrent_diarize.py` with the required arguments:

```bash
python whisper_diarization/concurrent_diarize.py -a audios/<audio_file.wav> --whisper-model <model_name>
```

### Command Line Arguments 
 
- `-a, --audio`: Name of the target audio file (required).
 
- `--no-stem`: Disables source separation. This helps with long files that don't contain a lot of music.
 
- `--suppress_numerals`: Suppresses numerical digits, improving diarization accuracy by converting all digits into written text.
 
- `--whisper-model`: Name of the Whisper model to use (default: `medium.en`).
 
- `--batch-size`: Batch size for batched inference. Reduce if you run out of memory; set to 0 for non-batched inference (default: 8).
 
- `--language`: Language spoken in the audio. Specify None to perform language detection.
 
- `--device`: Device to run the model on. Use `cuda` if you have a GPU, otherwise `cpu`.

### Script Overview 
`concurrent_diarize.py`
This script orchestrates the entire process of audio transcription and speaker diarization. Below is a high-level overview of the steps involved:
 
1. **Parsing Command Line Arguments** : The script accepts various arguments to customize the transcription and diarization process.
 
2. **Vocal Isolation** : Uses Demucs to separate vocals from background music if the `--no-stem` flag is not set.
 
3. **Transcription** : Utilizes the Whisper model for audio transcription.
 
4. **Forced Alignment** : Aligns the transcribed text with the audio using Wav2Vec2.
 
5. **Mono Audio Conversion** : Converts audio to mono for compatibility with NeMo MSDD.
 
6. **Speaker Diarization** : Performs speaker diarization using the NeMo MSDD model.
 
7. **Restoring Punctuation** : Restores punctuation in the transcribed text using a deep learning model.
 
8. **Writing Output Files** : Generates and saves the final speaker-aware transcript and SRT files.

The code runs Whisper and the NeMo MSDD model in concurrence for better efficiency.

## Configurations 
Configuration files for different diarization scenarios (general, meeting, telephonic) are stored in the `config` directory. You can customize these YAML files based on your specific needs.
## Output 
Processed outputs, including transcribed text files and SRT subtitle files, are saved in the `outputs` directory.
## Troubleshooting 

- Ensure your audio files are in a supported format (e.g., WAV).

- Verify that you have the correct versions of all dependencies installed.

- For CUDA-related issues, make sure your GPU drivers and CUDA toolkit are correctly installed.

## License 
This project is under `MIT License`. For more information, see the `LICENSE` file.