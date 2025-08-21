"""
Core compression artifact analyzer based on research paper methodology
Adapted for both music and network codecs
"""

import numpy as np
from scipy import signal
from scipy.fftpack import dct, idct
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class CompressionAnalyzer:
    """
    Analyzes audio for compression artifacts using the methodology from:
    "Lossy Audio Compression Identification" by Kim & Rafii
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.eps = 1e-10  # Small value to avoid log(0)
        
    def compute_mdct(self, x: np.ndarray, N: int) -> np.ndarray:
        """
        Compute Modified Discrete Cosine Transform
        Equation (1) from the paper
        """
        n = np.arange(N)
        k = np.arange(N // 2)
        
        # Create transform matrix
        transform_matrix = np.cos(
            (2 * np.pi / N) * np.outer(n + 0.5 + N/4, k + 0.5)
        )
        
        # Apply transform
        X = np.dot(x, transform_matrix)
        return X
    
    def compute_imdct(self, X: np.ndarray, N: int) -> np.ndarray:
        """Inverse MDCT"""
        k = np.arange(len(X))
        n = np.arange(N)
        
        # Create inverse transform matrix
        transform_matrix = (2.0 / N) * np.cos(
            (2 * np.pi / N) * np.outer(k + 0.5, n + 0.5 + N/4)
        )
        
        # Apply inverse transform
        x = np.dot(X, transform_matrix)
        return x
    
    def create_sine_window(self, N: int) -> np.ndarray:
        """
        Create sine window - Equation (2) from paper
        """
        n = np.arange(N)
        return np.sin(np.pi / N * (n + 0.5))
    
    def create_slope_window(self, N: int) -> np.ndarray:
        """
        Create slope window - Equation (3) from paper
        Used by Vorbis
        """
        n = np.arange(N)
        return np.sin(np.pi / 2 * np.sin(np.pi / N * (n + 0.5)) ** 2)
    
    def create_kbd_window(self, N: int, alpha: float = 4) -> np.ndarray:
        """
        Create Kaiser-Bessel Derived window - Equation (4) from paper
        Used by AAC and AC-3
        """
        from scipy.special import i0
        
        # Create Kaiser window
        n = np.arange(N)
        kaiser = np.zeros(N)
        
        for i in range(N // 2):
            num = i0(np.pi * alpha * np.sqrt(1 - (2*i/N - 1)**2))
            den = i0(np.pi * alpha)
            kaiser[i] = num / den
            kaiser[N-1-i] = kaiser[i]
        
        # Derive KBD window from Kaiser
        w = np.zeros(N)
        
        # First half
        for n in range(N // 2):
            w[n] = np.sqrt(np.sum(kaiser[:n+1]) / np.sum(kaiser[:N//2+1]))
        
        # Second half (symmetric)
        for n in range(N // 2, N):
            w[n] = np.sqrt(np.sum(kaiser[:N-n]) / np.sum(kaiser[:N//2+1]))
        
        return w
    
    def compute_spectrogram_mdct(self, audio: np.ndarray, window_size: int, 
                                 hop_size: int, window_type: str = "sine",
                                 window_params: dict = {}) -> np.ndarray:
        """
        Compute MDCT-based spectrogram for compression analysis
        """
        # Select window function
        if window_type == "sine":
            window = self.create_sine_window(window_size)
        elif window_type == "kbd":
            alpha = window_params.get("alpha", 4)
            window = self.create_kbd_window(window_size, alpha)
        elif window_type == "slope":
            window = self.create_slope_window(window_size)
        else:
            window = np.ones(window_size)  # Rectangular
        
        # Calculate number of frames
        num_frames = (len(audio) - window_size) // hop_size + 1
        
        # Initialize spectrogram
        spectrogram = np.zeros((window_size // 2, num_frames))
        
        # Compute MDCT for each frame
        for i in range(num_frames):
            start = i * hop_size
            frame = audio[start:start + window_size]
            
            if len(frame) < window_size:
                frame = np.pad(frame, (0, window_size - len(frame)))
            
            # Apply window
            windowed_frame = frame * window
            
            # Compute MDCT
            mdct_coeffs = self.compute_mdct(windowed_frame, window_size)
            spectrogram[:, i] = np.abs(mdct_coeffs)
        
        return spectrogram
    
    def compute_lp_residual(self, audio: np.ndarray, frame_size: int, 
                           lp_order: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LP residual for ACELP-based codecs (AMR, EVS, etc.)
        """
        from scipy.signal import lfilter
        from scipy.linalg import toeplitz
        
        num_frames = len(audio) // frame_size
        residuals = []
        lp_coeffs_all = []
        
        for i in range(num_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            
            if len(frame) < frame_size:
                continue
            
            # Compute autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(frame)-1:len(frame)-1+lp_order+1]
            
            # Levinson-Durbin recursion for LP coefficients
            if autocorr[0] != 0:
                # Create Toeplitz matrix
                R = toeplitz(autocorr[:lp_order])
                r = autocorr[1:lp_order+1]
                
                try:
                    # Solve for LP coefficients
                    lp_coeffs = np.linalg.solve(R, -r)
                    lp_coeffs = np.concatenate(([1], lp_coeffs))
                    
                    # Compute residual
                    residual = lfilter(lp_coeffs, 1, frame)
                    
                    residuals.append(residual)
                    lp_coeffs_all.append(lp_coeffs)
                except:
                    residuals.append(np.zeros(frame_size))
                    lp_coeffs_all.append(np.zeros(lp_order + 1))
            else:
                residuals.append(np.zeros(frame_size))
                lp_coeffs_all.append(np.zeros(lp_order + 1))
        
        return np.array(residuals), np.array(lp_coeffs_all)
    
    def detect_compression_traces(self, spectrogram: np.ndarray, 
                                 threshold: float = 0.01) -> Dict[str, float]:
        """
        Detect traces of compression in spectrogram
        Based on zeroed coefficients and energy patterns
        """
        # Convert to dB scale
        spectrogram_db = 20 * np.log10(spectrogram + self.eps)
        
        # Compute average energy for each frame
        frame_energies = np.mean(spectrogram_db, axis=0)
        
        # Compute differences between successive frames
        energy_diffs = np.diff(frame_energies)
        
        # Keep only positive differences (energy drops)
        positive_diffs = np.maximum(energy_diffs, 0)
        
        # Find peaks in differences (compression artifacts)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(positive_diffs, height=threshold)
        
        # Calculate metrics
        metrics = {
            'max_diff': np.max(positive_diffs) if len(positive_diffs) > 0 else 0,
            'mean_diff': np.mean(positive_diffs),
            'std_diff': np.std(positive_diffs),
            'num_peaks': len(peaks),
            'peak_regularity': 0  # Will be calculated next
        }
        
        # Calculate peak regularity (periodic peaks indicate correct codec)
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            if len(peak_intervals) > 0:
                metrics['peak_regularity'] = 1.0 / (1.0 + np.std(peak_intervals))
        
        # Count zero/near-zero coefficients
        zero_ratio = np.sum(spectrogram < threshold) / spectrogram.size
        metrics['zero_ratio'] = zero_ratio
        
        # High-frequency cut detection
        freq_energies = np.mean(spectrogram, axis=1)
        cutoff_idx = self._find_frequency_cutoff(freq_energies)
        metrics['freq_cutoff'] = cutoff_idx / len(freq_energies)
        
        return metrics
    
    def _find_frequency_cutoff(self, freq_energies: np.ndarray, 
                               threshold_ratio: float = 0.01) -> int:
        """Find frequency cutoff point (common in lossy compression)"""
        max_energy = np.max(freq_energies)
        threshold = max_energy * threshold_ratio
        
        # Find last frequency bin with significant energy
        for i in range(len(freq_energies) - 1, -1, -1):
            if freq_energies[i] > threshold:
                return i
        return 0
    
    def analyze_g711_artifacts(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Specific analysis for G.711 μ-law/A-law compression
        Detects quantization patterns from companding
        """
        # Compute histogram of sample values
        hist, bins = np.histogram(audio, bins=256)
        
        # G.711 has non-uniform quantization - detect this pattern
        # Calculate entropy of histogram
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + self.eps))
        
        # Detect companding curve artifacts
        # μ-law and A-law have specific quantization patterns
        diff = np.diff(audio)
        quantization_score = 1.0 / (1.0 + np.std(diff))
        
        return {
            'entropy': entropy,
            'quantization_score': quantization_score,
            'dynamic_range': np.max(audio) - np.min(audio)
        }
    
    def analyze_subband_artifacts(self, audio: np.ndarray, 
                                  cutoff_freq: int = 4000) -> Dict[str, float]:
        """
        Specific analysis for sub-band codecs like G.722
        """
        # Apply QMF filter bank
        nyquist = self.sample_rate // 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Design QMF filters
        b_low, a_low = signal.butter(8, normalized_cutoff, 'low')
        b_high, a_high = signal.butter(8, normalized_cutoff, 'high')
        
        # Split into sub-bands
        low_band = signal.filtfilt(b_low, a_low, audio)
        high_band = signal.filtfilt(b_high, a_high, audio)
        
        # Analyze each sub-band for ADPCM artifacts
        low_energy = np.mean(low_band ** 2)
        high_energy = np.mean(high_band ** 2)
        
        # G.722 allocates different bits to each band
        energy_ratio = high_energy / (low_energy + self.eps)
        
        return {
            'subband_energy_ratio': energy_ratio,
            'low_band_variance': np.var(low_band),
            'high_band_variance': np.var(high_band)
        }
    
    def combine_block_estimates(self, scores_list: List[Dict[str, float]], 
                               indices_list: List[int], 
                               period: int) -> float:
        """
        Combine estimates from multiple blocks using circular mean
        As described in Section III-B of the paper
        """
        if len(scores_list) == 0:
            return 0
        
        # Convert to polar coordinates
        radii = []
        angles = []
        
        for score_dict, idx in zip(scores_list, indices_list):
            # Use max_diff as the main score
            radius = score_dict.get('max_diff', 0)
            angle = (2 * np.pi * idx) / period  # Map index to angle
            
            radii.append(radius)
            angles.append(angle)
        
        # Convert to Cartesian coordinates
        x_coords = [r * np.cos(a) for r, a in zip(radii, angles)]
        y_coords = [r * np.sin(a) for r, a in zip(radii, angles)]
        
        # Compute circular mean
        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)
        
        # Convert back to polar
        final_score = np.sqrt(mean_x**2 + mean_y**2)
        
        return final_score