import streamlit as st
import argparse
import os
import io
import torch
from colorama import Fore, Style, init
from dotenv import load_dotenv
from datetime import datetime
from audio_processing import convert_audio_to_wav, isolate_vocals, save_audio_mono
from diarization import perform_diarization, read_speaker_timestamps, restore_punctuation
from helpers import (cleanup, get_realigned_ws_mapping_with_punctuation, get_sentences_speaker_mapping,
                     write_output_files, get_words_speaker_mapping, WHISPER_LANGS)
from transcription import transcribe_audio, perform_forced_alignment
from contextlib import redirect_stdout, redirect_stderr

load_dotenv()
init(autoreset=True)

# Create a logger class to capture print statements and redirect them to Streamlit
class StreamlitLogger(io.StringIO):
    def __init__(self, log_container):
        super().__init__()
        self.log_container = log_container  # Store the container to update logs

    def write(self, message):
        # Append the message to the Streamlit log box
        log_value = self.getvalue() + message
        # Update log box with styles to make it look like a console
        self.log_container.markdown(
            f"<div style='height:200px; overflow-y:auto; background-color:#f1f1f1; color:gray; font-family:monospace; font-size:12px;'>{log_value}</div>",
            unsafe_allow_html=True
        )
        super().write(message)

    def flush(self):
        pass  # This is needed to support Python 3

def main():
    # Streamlit interface
    st.title("Audio Transcription and Diarization Tool")

    # File upload
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac"])

    # Select parameters
    stem = st.checkbox("Enable Source Separation (Stemming)", value=True)
    suppress_numerals = st.checkbox("Suppress Numerical Digits", value=False)
    model_name = st.selectbox("Select Whisper Model", ["tiny", "base", "small", "medium", "large"])
    batch_size = st.number_input("Batch Size", min_value=0, max_value=64, value=8)
    language = st.selectbox("Language (leave empty for auto-detection)", [""] + WHISPER_LANGS)
    device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

    # Create a text area for logs at the bottom of the page
    log_box = st.empty()  # This creates an empty Streamlit container for the logs

    # Run the processing on button click
    if st.button("Run"):
        if audio_file is not None:
            # Use a temporary file to save the uploaded file
            temp_audio_path = os.path.join("temp_inputs", audio_file.name)
            os.makedirs("temp_inputs", exist_ok=True)
            with open(temp_audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            # Create argument parser namespace
            args = argparse.Namespace(
                audio=temp_audio_path,
                stemming=stem,
                suppress_numerals=suppress_numerals,
                model_name=model_name,
                batch_size=batch_size,
                language=language if language else None,
                device=device
            )

            # Redirect standard output and error to the logger
            logger = StreamlitLogger(log_box)
            with redirect_stdout(logger), redirect_stderr(logger):
                # Run the audio processing
                run_audio_processing(args)
        else:
            st.error("Please upload a valid audio file.")

    # Display a footer with your name at the bottom
    st.markdown("<p style='text-align: center; margin-top: 50px;'>Made by Thibaud Bourgeois</p>", unsafe_allow_html=True)

def run_audio_processing(args):
    """
    Runs the entire audio transcription and diarization process with provided arguments.
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}Starting process...{Style.RESET_ALL}")

    # Convert audio format to .wav and isolate vocals from the audio file if stemming is enabled
    converted_audio = convert_audio_to_wav(args.audio)
    vocal_target = isolate_vocals(converted_audio, args.stemming)

    # Transcription and diarization (avoid concurrent execution if logs update Streamlit elements)
    whisper_results, language, audio_waveform = transcribe_audio(vocal_target, args)

    # Save the audio in mono format for diarization
    temp_path = os.path.join(os.getcwd(), "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    save_audio_mono(audio_waveform, temp_path)

    # Perform diarization
    perform_diarization(temp_path, args)

    # Perform forced alignment of the transcribed text with the audio
    word_timestamps = perform_forced_alignment(whisper_results, language, audio_waveform, args)

    # Read diarization results and process them
    speaker_ts = read_speaker_timestamps(temp_path)
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    wsm = restore_punctuation(wsm, language)
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Write the output files and retrieve their paths
    txt_filename, srt_filename = write_output_files(ssm, args.model_name)

    # Clean up temporary files
    cleanup(temp_path)
    os.remove(args.audio)

    # Display completion message and output file
    print(f"{Fore.CYAN}{Style.BRIGHT}Process completed successfully!{Style.RESET_ALL}")

    # Display the output SRT file content
    with open(srt_filename, "r") as output_file:
        st.text_area("Output SRT File", output_file.read(), height=300, key="output_srt")


if __name__ == "__main__":
    main()
