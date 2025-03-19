## General Imports
import numpy as np

## Visualization
from matplotlib import pyplot as plt

## Audio Imports
import librosa, librosa.display           
from music21.tempo import MetronomeMark   
from music21.note import Note, Rest



class Analysis:
    def __init__(self, signal, sample_rate, n_bins=72, mag_exp=4, pre_post_max=6, threshold=-70):
        self.signal = signal
        self.sr = sample_rate
        self.hop_length = int(sample_rate * 0.01)                # Number of samples between successive frames
        self.n_bins = n_bins                            # Number of frequency bins
        self.mag_exp = mag_exp                          # Magnitude Exponent
        self.pre_post_max = pre_post_max                # Pre- and post- samples for peak picking
        self.threshold = threshold  
        self.get_cqt(signal)

    def get_cqt(self, signal):
        cqt = librosa.cqt(signal, sr=self.sr, hop_length=self.hop_length, fmin=None, n_bins=self.n_bins)
        cqt_mag = librosa.magphase(cqt)[0]**self.mag_exp
        cqt_dB = librosa.core.amplitude_to_db(cqt_mag ,ref=np.max)
        self.cqt = cqt_dB
    
    def cqt_thresholded(self,thres=None):
        if thres is None:
            thres = self.threshold
        #sets every value under the threshold to -120 dB
        new_cqt=np.copy(self.cqt)
        new_cqt[new_cqt<thres]=-120
        self.cqt = new_cqt
    
    #Onset detection - when does a musical note start
    def calc_onset(self, backtrack=True):
        onset_env = librosa.onset.onset_strength(S=self.cqt,sr=self.sr, hop_length=self.hop_length)

        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length, backtrack=backtrack,pre_max=self.pre_post_max,
                                           post_max=self.pre_post_max)

        onset_boundaries = np.concatenate([[0], onset_frames, [self.cqt.shape[1]]])
        onset_times = librosa.frames_to_time(onset_boundaries, sr=self.sr, hop_length=self.hop_length)
        self.onsets = [onset_times, onset_boundaries, onset_env]
            
    #Estimating pitch and notes
    def est_tempo(self):
        tempo, _=librosa.beat.beat_track(y=None, sr=self.sr, onset_envelope=self.onsets[2], hop_length=self.hop_length,
                start_bpm=100.0, tightness=100, trim=True, bpm=None,
                units='frames')
        tempo=int(2*round(tempo[0]/2))
        self.mm = MetronomeMark(referent='quarter', number=tempo) #quater notes instead of time
        self.tempo = tempo
        print(f"est tempo: {tempo}")
    
    def time_to_beat(self, duration, tempo):
        return (tempo*duration/60)
    
    # Remap input to 0-1 for Sine Amplitude or to 0-127 for MIDI
    def remap(self, f0, in_min, in_max, out_min, out_max):
        return (f0 - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    # Generate Sinewave, MIDI Notes and music21 notes
    def generate_sine_midi_note(self, f0_info, sr, n_duration, round_to_sixtenth=True):
        f0=f0_info[0]
        A=self.remap(f0_info[1], self.cqt.min(), self.cqt.max(), 0, 1)
        duration = librosa.frames_to_time(n_duration, sr=self.sr, hop_length=self.hop_length)
        #Generate Midi Note and music21 note
        note_duration = round(0.02*np.around(duration/2/0.02),2) # Round to 2 decimal places for music21 compatibility
        midi_duration = self.time_to_beat(duration, self.tempo)
        midi_velocity=int(round(self.remap(f0_info[1], self.cqt.min(), self.cqt.max(), 0, 127)))
       
        if note_duration == 0:
            empty_wave = np.zeros(1)  # Placeholder for consistency
            empty_midi_info = [None, 0, 0]  # MIDI placeholder
            empty_note = [Rest()]  # Wrap Rest() in a list to match expected structure

            return [empty_wave, empty_midi_info, empty_note]
        print(self.mm.secondsToDuration(note_duration).type)
        if self.mm.secondsToDuration(note_duration).type == 'inexpressible':
            beat_duration = "16th"
        else:
            beat_duration = self.mm.secondsToDuration(note_duration).type

        if round_to_sixtenth:
            midi_duration=round(midi_duration*16)/16

        if f0==None: 
            midi_note=None
            note_info=Rest(type=beat_duration)
            f0=0
        else:
            midi_note=round(librosa.hz_to_midi(f0))
            note_name = librosa.midi_to_note(midi_note).replace("♯", "#")
            note = Note(note_name, type=beat_duration)
            note.volume.velocity = 50 #midi_velocity
            note_info = [note]

        midi_info = [midi_note, midi_duration, midi_velocity]
                
        # Generate Sinewave
        n = np.arange(librosa.frames_to_samples(n_duration, hop_length=self.hop_length ))
        sine_wave = A*np.sin(2*np.pi*f0*n/float(sr))
        
        return [sine_wave, midi_info, note_info]
    
    #Estimate Pitch
    def estimate_pitch(self, segment):
        freqs = librosa.cqt_frequencies(n_bins=self.n_bins, fmin=librosa.note_to_hz('C1'),
                                bins_per_octave=12)
        if segment.max()<self.threshold:
            return [None, np.mean((np.amax(segment,axis=0)))]
        else:
            f0 = int(np.mean((np.argmax(segment,axis=0))))
        print(librosa.hz_to_note(freqs[f0]))
        return [freqs[f0], np.mean((np.amax(segment,axis=0)))]
    
    # Generate notes from Pitch estimation
    def estimate_pitch_and_notes(self, i):
        onset_boundaries = self.onsets[1]
        n0 = onset_boundaries[i]
        n1 = onset_boundaries[i+1]
        f0_info = self.estimate_pitch(np.mean(self.cqt[:,n0:n1],axis=1))
        return self.generate_sine_midi_note(f0_info, self.sr, n1-n0)

    def show_spect(self,Y=None):
        if Y is None:
            Y = self.cqt
        librosa.display.specshow(Y, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='cqt_note', cmap='coolwarm')
        plt.ylim([librosa.note_to_hz('B2'),librosa.note_to_hz('B6')])

        plt.vlines(self.onsets[0], 0, self.sr/2, color='k', alpha=0.8)
        plt.title("CQTg")
        plt.colorbar(format='%+2.0f dB')
        plt.show()

















    