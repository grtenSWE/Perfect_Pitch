from IPython.utils.io import warn
import numpy as np # numpy for working with matrices
import scipy.signal

import librosa
from matplotlib import pyplot as plt # plotting package



class Preprocess:
    def __init__(self, filename, start_t=0, end_t=None, n_bins=72, mag_exp=4):
        self.signal, self.sr = self.read_wav(filename, start_t, end_t)
        self.apply_hpss()                               # Separate harmonics & remove percussive noise
        self.apply_dynamic_compression()
        self.mags, self.freqs = self.time_to_freq()

        self.hop_length = int(self.sr * 0.01)           # Number of samples between successive frames
        self.n_bins = n_bins                            # Number of frequency bins
        self.mag_exp = mag_exp                          # Magnitude Exponent

    def read_wav(self, filename, start_t, end_t):
        signal, sr = librosa.load(filename, sr=None, mono=True, offset=start_t, duration=end_t)
        signal, _ = librosa.effects.trim(signal)
        return signal, sr

    def apply_hpss(self):
        """ Harmonic-Percussive Source Separation to keep only harmonic content. """
        self.signal, _ = librosa.effects.hpss(self.signal)

    def apply_dynamic_compression(self):
        """ Applies dynamic compression while handling outlier loud notes. """
        rms = librosa.feature.rms(y=self.signal, frame_length=2048, hop_length=512)[0]  

        # compute median RMS and standard deviation
        median_rms = np.median(rms)
        rms_std = np.std(rms)
        loud_threshold = median_rms + 2.5 * rms_std  

        # Filter for weak and loud indices 
        gain = np.ones_like(rms)
        weak_indices = rms < median_rms
        loud_indices = rms > loud_threshold

        # Scale only weak signals and reduce loud ones
        gain[weak_indices] = 50*(median_rms / (rms[weak_indices] + 0.001))
        gain[loud_indices] = loud_threshold / rms[loud_indices]  

        # Prevent extreme changes and smooth gain function
        gain = np.clip(gain, 0.5, 1.5)
        gain = scipy.signal.medfilt(gain, kernel_size=5)

        # Interpolate gain to match signal length
        gain_interp = np.interp(
            np.linspace(0, len(self.signal), num=len(self.signal)),  
            np.linspace(0, len(self.signal), num=len(gain)),  
            gain  
        )

        self.signal = self.signal * gain_interp

    def make_symmetric(self, Y):
        """
        Ensures the frequency spectrum is symmetric for real-valued time-domain reconstruction.
        Only needed if you zeroed out or modified one side of the FFT.
        """
        N = len(Y)
        # Preserve DC and Nyquist, and mirror the positive freqs to negative
        Y_sym = np.zeros_like(Y, dtype=np.complex128)
        Y_sym[0] = Y[0]  # DC component
        Y_sym[1:N//2] = Y[1:N//2]
        Y_sym[N//2+1:] = np.conj(Y[1:N//2][::-1])
        if N % 2 == 0:  # Nyquist freq for even-length
            Y_sym[N//2] = Y[N//2]
        return Y_sym

    def time_to_freq(self):
        """ Returns the fourier transform of a signal as well as the corresponding frequencies """
        # In the context of dictionary learning, this is returning the sparse representation matrix A

        ###NEED TO FIX THE ISSUE OF INFO LOSS DURING TRANFORMATION
        n = len(self.signal)
        # Take the Fourier transform
        Y_full = np.fft.fft(self.signal)
        freq_full = np.fft.fftfreq(n, d=1/self.sr)
        return Y_full, freq_full
    
    def freq_to_time(self, Y=None):
        """ Reconstructs the data after denoising the representation """
        print("converting to back to signal...")
        if Y is None:
            Y = self.mags

        #Y = self.make_symmetric(Y)
        signal = np.fft.ifft(Y)
        complex_mag = np.linalg.norm(np.imag(signal))
        if complex_mag > 0.01 * np.linalg.norm(signal):
            warn("There is a large complex part to the time domain signal")
        signal = np.real(signal)
        return signal
    
    def get_cqt(self, signal):
        cqt = librosa.cqt(signal, sr=self.sr, hop_length=self.hop_length, n_bins=self.n_bins)
        cqt_mag = librosa.magphase(cqt)[0]**self.mag_exp
        cqt_dB = librosa.core.amplitude_to_db(cqt_mag ,ref=np.max)
        self.cqt = cqt_dB

    # -- Frequency filtering methods --
    def keep_top_n_freqs(self, n=200, window=1):
        """ Keeps only the top n frequencies within a specified frequency window, setting all other frequencies to zero to simplify the signal's frequency content.
        This makes the representation sparse again by keeping the most dominant components. Since noise is spreadout in the frequency domain, it is removed by a large extent. """
        Y = np.empty_like(self.mags)  
        np.copyto(Y, self.mags)  
        
        Y_new = np.zeros(len(Y), dtype=np.complex128)
        for _ in range(n):
            pos_indices = np.where(self.freqs > 0)[0]  # Get indices of positive frequencies
            i_max = pos_indices[np.argmax(np.abs(Y[pos_indices]))]  #get indicies of max frequency
            f_max = self.freqs[i_max] #get the frequency of that largest fourier transform magnitude
            
            #prevents Y_new from being over written with zeros. if i_max returns 1, that means Y is most likely to be zeroed out. 
            if i_max == 1: 
                return Y_new
            
            pos_mask = np.bitwise_and(self.freqs  >= (f_max-window), self.freqs  <= (f_max+window)) #inds_window_pos is a boolean array of the and of these 2 conditions
            neg_mask = np.bitwise_and(self.freqs  >= (-f_max-window), self.freqs  <= (-f_max+window))
            full_mask = np.bitwise_or(pos_mask, neg_mask) #combining the boolean array for negative and positive freqs to get the pair. 

            Y_new[full_mask] = self.mags[full_mask]
            Y[full_mask] = 0  

        return Y_new
    
    def keep_top_freqs(self,mag_min=1000):
        """ Keeps only the frequencies above a certain fourier transform magnitude threshold """
        Y_new = np.zeros(len(self.mags), dtype=np.complex128)
        idx = np.abs(self.mags) >= mag_min
        Y_new[idx] = self.mags[idx]
        return Y_new
   
    def remove_unwanted_freqs(self, f_min=27.5, f_max=4186):
        """ Keeps only frequencies within a range"""
        if type(f_max) is str:
            f_max = librosa.note_to_hz(f_max)

        if type(f_min) is str:
            f_min = librosa.note_to_hz(f_min)

        Y_new = np.zeros(len(self.mags), dtype=np.complex128)
        bool_idx_pos = np.bitwise_and(self.freqs >= f_min, self.freqs <= f_max)
        bool_idx_neg = np.bitwise_and(self.freqs <= -f_min, self.freqs >= -f_max)
        bool_idx = np.bitwise_or(bool_idx_pos,bool_idx_neg)
        Y_new[bool_idx] = self.mags[bool_idx]
        self.mags = Y_new
    
    def auto_denoise(self):
        self.plot_freq()
        magnitude_spectrum = np.abs(self.mags)

        # Check if signal has a few dominant frequencies
        top_freq_magnitudes = np.sort(magnitude_spectrum[self.freqs > 0])[-20:]  # Top 20 frequencies
        mean_top_freq = np.mean(top_freq_magnitudes)

        if mean_top_freq > 0.1 * np.max(magnitude_spectrum):  # Few strong peaks. If the mean of the top 10 is at least 10% of the max, then there isn't a single high peak.
            print("Applying top-N frequency selection...")
            n = self.find_best_n_peaks()
            print(n)
            self.mags = self.keep_top_n_freqs(n,30)  # Keep n most dominant frequencies
        
        print("Applying magnitude-based filtering...")
        self.mags = self.keep_top_freqs(0.01*mean_top_freq)  # Remove weak frequencies
        self.plot_freq()

    def find_best_n(self, energy_threshold=0.9):
        #Finds the optimal n such that the top n frequencies retain the given energy threshold.
        magnitude_spectrum = np.abs(self.mags)
        sorted_magnitudes = np.sort(magnitude_spectrum[self.freqs > 0])[::-1]  # Sort in descending order
        
        total_energy = np.sum(sorted_magnitudes)  # Total energy in the spectrum
        cumulative_energy = np.cumsum(sorted_magnitudes)  # Cumulative sum of sorted magnitudes
        
        # Find the smallest n where cumulative energy exceeds the threshold
        n_optimal = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
        return n_optimal
    
    def find_best_n_peaks(self, prominence_ratio=0.15):
        #Uses peak detection to determine the number of strong frequencies to keep.
        magnitude_spectrum = np.abs(self.mags)
        
        # Detect peaks in the magnitude spectrum
        peaks, _ = scipy.signal.find_peaks(magnitude_spectrum, prominence=prominence_ratio * np.max(magnitude_spectrum))
        
        return min(len(peaks), 10000)  # Cap at 10000

    # -- plotting methods --
    def plot_freq(self, mags=None):
        if mags is None:
            mags = self.mags

        # Plot only the magnitude of the FT for only positive frequencies
        # (abstract away negative frequencies and compelex numbers)
        range_bool_idx = np.bitwise_and(self.freqs >=0, self.freqs < 6000)

        plt.figure(figsize=(10,5))
        plt.plot(self.freqs[range_bool_idx], np.abs(mags[range_bool_idx]))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Fourier transform magnitude")
        plt.title("Frequency Spectrum")
        plt.grid()
        plt.show()

    def plot_signal(self):
        # plot the signal
        ts = np.arange(0, len(self.signal)/self.sr, 1/self.sr) # time values for sampling
        plt.plot(ts, self.signal)
        plt.xlabel("time (second)")
        plt.ylabel("Input signal y")
        plt.show()

    def show_spect(self,Y=None):
        if Y is None:
            Y = self.cqt
        librosa.display.specshow(Y, sr=self.sr, hop_length=int(self.sr * 0.01) , x_axis='time', y_axis='cqt_note', cmap='coolwarm')
        plt.ylim([librosa.note_to_hz('B2'),librosa.note_to_hz('B6')])
        plt.title("CQTg")
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    def print_top_n(self,n=10, mags=None):
        if mags is None:
            mags = self.mags

        top_indices = np.argsort(np.abs(mags))[-n:]  # Get indices of top 10 peaks
        for i in top_indices:
            print(f"Freq: {self.freqs[i]:.2f} Hz, Magnitude: {np.abs(mags[i]):.2f}")
 
 











