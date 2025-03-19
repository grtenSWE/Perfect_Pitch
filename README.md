Monophonic Piano Transcription using Signal Processing 🎹

Overview

This project converts monophonic piano recordings into sheet music using signal processing techniques. By analyzing audio frequency components, it identifies and transcribes musical notes. This implementation focuses on single-note recognition and lays the groundwork for future enhancements, including polyphonic transcription and machine learning integration.

Features

✅ Audio Preprocessing – Noise reduction and normalization for clearer signal analysis.
✅ Fourier Transform-Based Note Detection – Uses Fast Fourier Transform (FFT) to extract dominant frequencies.
✅ Peak Detection Algorithm – Identifies musical notes based on frequency peaks.
✅ MIDI Output Generation – Converts detected notes into MIDI format for playback.
✅ End-to-End Pipeline – From raw audio input to sheet music representation.


Usage
  1.	Place your monophonic piano audio file (WAV format recommended) in the input/ directory.
	2.	Run the transcription script:  
           python main.py <filename> --start_t <start_time> --end_t <end_time> --f_min <min_freq> --f_max <max_freq>
  3.	The detected notes will be saved as a MIDI file and a wav file in the output/ directory.


Technologies Used
	•	Python 🐍
	•	NumPy & SciPy – Signal processing and spectral analysis
	•	Matplotlib – Visualizing audio waveforms and frequency spectrums
	•	MIDI Libraries – Generating playable sheet music

Limitations & Future Improvements

🚧 Current Limitations
	•	Only supports monophonic (single-note) piano recordings.
	•	Accuracy may vary with background noise and note overlap.

🚀 Planned Enhancements
	•	Implement polyphonic transcription to handle chords.
	•	Explore machine learning models for improved note recognition.
	•	Support for real-time audio transcription.
