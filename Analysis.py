## General Imports
import numpy as np

## Audio Imports
import librosa, librosa.display           
from music21.tempo import MetronomeMark   

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH



class Analysis:
    def __init__(self, filepath, cqt, sample_rate):
        self.filepath = filepath
        self.sr = sample_rate
        self.hop_length = int(sample_rate * 0.01)   # Number of samples between successive frames
        self.cqt = cqt
        self.calc_onset()

    def get_midi(self):
        model_output, midi_data, note_events = predict(
            audio_path=self.filepath,
            #onset_threshold=self.estimate_onset_threshold(),
            #frame_threshold=self.estimate_frame_threshold(),
            #midi_tempo=self.estimate_tempo(),
        )
        return model_output, midi_data, note_events
    
    def estimate_onset_threshold(self):
        """
        Estimate an onset threshold based on the onset energy envelope.
        """
        thresh = np.mean(self.onset_env) + np.std(self.onset_env)
        normalized_thresh = thresh / np.max(self.onset_env)
        return normalized_thresh

    def estimate_frame_threshold(self):
        """
        Estimate a frame threshold from the CQT energy.
        """
        mean_energy = np.mean(self.cqt, axis=1)
        thresh = np.percentile(mean_energy, 50)  # or adjust percentile as needed
        normalized_thresh = thresh / np.max(self.cqt)
        return normalized_thresh

    
    #Onset detection - when does a musical note start
    def calc_onset(self, backtrack=True):
        self.onset_env = librosa.onset.onset_strength(S=self.cqt,sr=self.sr, hop_length=self.hop_length)
            
    #Estimating pitch and notes
    def estimate_tempo(self):
        tempo, _=librosa.beat.beat_track(y=None, sr=self.sr, onset_envelope=self.onset_env, hop_length=self.hop_length,
                start_bpm=100.0, tightness=100, trim=True, bpm=None,
                units='frames')
        tempo=int(2*round(tempo[0]/2))
        self.mm = MetronomeMark(referent='quarter', number=tempo) #quater notes instead of time
        print(f"est tempo: {tempo}")
        return tempo
    
















    