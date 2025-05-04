🎹 Polyphonic Piano Transcription using Signal Processing

Overview

This project converts piano recordings into sheet music and MIDI files by combining signal processing techniques with a deep learning model. It analyzes the frequency content of audio to identify and transcribe musical notes. Initially built for monophonic transcription, the system now supports polyphonic recordings using Spotify’s Basic Pitch, a lightweight neural network for pitch and onset detection.

⸻

Features

✅ Audio Preprocessing – Applies harmonic-percussive separation, dynamic compression, and frequency filtering to enhance signal quality.
✅ Fourier Transform-Based Analysis – Extracts dominant frequencies using FFT to enable denoising and spectral analysis.
✅ Peak Detection & Spectral Denoising – Filters high-energy frequencies to isolate harmonic content and suppress noise.
✅ Deep Learning-Based Transcription – Leverages Spotify’s Basic Pitch model to extract onsets, pitches, and durations for polyphonic transcription.
✅ MIDI Output Generation – Converts detected notes into playable MIDI files using music21.
✅ End-to-End Pipeline – Fully automated workflow from raw audio input to MIDI and synthesised waveform output.

⸻

Usage
  1.	Place your piano audio file (WAV format recommended) in the input/ directory.
	2.	Run the transcription script:  
           python main.py <filename> --start_t <start_time> --end_t <end_time> --f_min <min_freq> --f_max <max_freq>
  3.	The detected notes will be saved as a MIDI file in the output/ directory.


Technologies Used
	•	Python 🐍
	•	Librosa – Audio loading, HPSS, and spectral transforms
	•	NumPy & SciPy – Signal processing and numerical operations
	•	Matplotlib – Spectrogram and waveform visualisation
	•	music21 – MIDI file generation and music representation
	•	Spotify’s Basic Pitch – Deep learning model for real-time audio-to-MIDI transcription

⸻

Limitations & Future Improvements

🚧 Current Limitations
	•	Accuracy may degrade with poor-quality recordings or overlapping background noise.
	•	MIDI dynamics and articulation are not yet included.

🚀 Planned Enhancements
	•	Add note confidence filtering and post-processing for cleaner outputs.
	•	Introduce real-time transcription and audio streaming capabilities.
	•	Explore fine-tuning Basic Pitch or augmenting it with rule-based logic for improved control.

⸻
