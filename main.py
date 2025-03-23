from Preprocess import Preprocess
from Analysis import Analysis

import argparse
import os
import numpy as np
from scipy.io import wavfile



def process_audio(filename, start_t=0, end_t=None, f_min="C1", f_max="C8"):
    """
    Preprocesses and analyzes an audio file.
    """
    try:
        print(f"Processing file: {filename}.wav")
        
        # Step 1: Preprocess audio
        audio_file = Preprocess(f"Music/{filename}.wav", start_t=start_t, end_t=end_t)
        audio_file.auto_denoise()
        audio_file.remove_unwanted_freqs(f_min=f_min, f_max=f_max)
        #audio_file.show_spect()
        signal = audio_file.freq_to_time()
        audio_file.get_cqt(signal)
        write_wav(f"{filename}_denoised.wav",signal,audio_file.sr)
        
        # Step 2: Analyze the processed signal
        analysis = Analysis(f"{filename}_denoised.wav", audio_file.cqt, audio_file.sr)
        music_info = analysis.get_midi()
        #os.remove(f"{filename}_denoised.wav")
        
        return music_info
    
    except FileNotFoundError:
        print(f"Error: File {filename}.wav not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def write_wav(filename, sig, sr):
    """ Saves a waveform as a WAV file. """
    scaled = np.int16(sig / np.max(np.abs(sig)) * np.iinfo(np.int16).max)
    wavfile.write(filename, sr, scaled)

def generate_midi(filename, midi_data):
    """ Generates a MIDI file from the extracted notes. """
    # Write to MIDI file
    midi_data.write(f'{filename}.mid')
    print(f"MIDI file saved: {filename}.mid")

def main():
    """
    Main execution function for processing audio files.
    """
    parser = argparse.ArgumentParser(description="Piano Music Transcription")
    parser.add_argument("filename", type=str, help="Name of the audio file (without .wav extension)")
    parser.add_argument("--start_t", type=float, default=0, help="Start time (in seconds)")
    parser.add_argument("--end_t", type=float, default=None, help="End time (in seconds)")
    parser.add_argument("--f_min", type=str, default="C1", help="Minimum frequency to retain")
    parser.add_argument("--f_max", type=str, default="C8", help="Maximum frequency to retain")
    
    args = parser.parse_args()
    
    music_info = process_audio(args.filename, args.start_t, args.end_t, args.f_min, args.f_max)
    
    if music_info:
        print("Analysis complete. Generating MIDI file...")
        generate_midi(args.filename, music_info[1])
    
if __name__ == "__main__":
    main()
