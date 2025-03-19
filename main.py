from Preprocess import Preprocess
from Analysis import Analysis

import argparse
import numpy as np
from scipy.io import wavfile

## Audio Imports
from music21 import metadata
from music21 import instrument
from music21.stream import Stream






def process_audio(filename, start_t=0, end_t=None, f_min="C1", f_max="C8"):
    """
    Preprocesses and analyzes an audio file.
    """
    try:
        print(f"Processing file: {filename}.wav")
        
        # Step 1: Preprocess audio
        audio_file = Preprocess(f"{filename}.wav", start_t=start_t, end_t=end_t)
        audio_file.auto_denoise()
        audio_file.remove_unwanted_freqs(f_min=f_min, f_max=f_max)
        signal = audio_file.freq_to_time()
        
        # Step 2: Analyze the processed signal
        analysis = Analysis(signal, audio_file.sr, pre_post_max=15, threshold=-80)
        analysis.cqt_thresholded()
        analysis.calc_onset()
        analysis.est_tempo()
        analysis.show_spect()
        
        return analysis
    
    except FileNotFoundError:
        print(f"Error: File {filename}.wav not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def write_wav(fname, sig, fs):
    """ Saves a waveform as a WAV file. """
    scaled = np.int16(sig / np.max(np.abs(sig)) * np.iinfo(np.int16).max)
    wavfile.write(fname, fs, scaled)

def generate_midi(filename, audio_file):
    """ Generates a MIDI file from the extracted notes. """
    music_info = [audio_file.estimate_pitch_and_notes(i) for i in range(len(audio_file.onsets[1])-1)]
    sine_waves = [item[0] for item in music_info]
    midi_notes = [item[1] for item in music_info]
    note_objects = [item[2] for item in music_info]
    
    # Concatenate synthesized sine wave audio
    synth_audio = np.concatenate(sine_waves)
    write_wav(f"{filename}_synth.wav", synth_audio, audio_file.sr)
    
    # Create a MIDI stream
    s = Stream()
    s.append(audio_file.mm)
    piano = instrument.fromString('Piano')
    piano.midiChannel = 0
    piano.midiProgram = 1  # Set to Acoustic Grand Piano
    s.append(piano)
    s.insert(0, metadata.Metadata())
    s.metadata.title = f"{filename} - Transcription"
    s.metadata.composer = "Andy"
    
    for note in note_objects:
        s.append(note)
    
    # Analyze key and insert into stream
    key = s.analyze('key')
    print(f"Detected Key: {key.name}")
    s.insert(0, key)
    
    # Write to MIDI file
    s.write('midi', f'{filename}.mid')
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
    
    analysis = process_audio(args.filename, args.start_t, args.end_t, args.f_min, args.f_max)
    
    if analysis:
        print("Analysis complete. Generating MIDI file...")
        generate_midi(args.filename, analysis)
    
if __name__ == "__main__":
    main()
