"""
Main codec identification system
Combines all analysis methods to identify codec with probability scores
"""

import numpy as np
import soundfile as sf
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import json

from src.codec_profiles import CODEC_PROFILES, CodecProfile, TransformType
from src.analyzers.compression_analyzer import CompressionAnalyzer

warnings.filterwarnings('ignore')

class CodecIdentifier:
    """
    Main class for identifying audio codec from WAV file
    Returns probability distribution over possible codecs
    """
    
    def __init__(self, block_duration: float = 1.0, verbose: bool = False):
        """
        Initialize codec identifier
        
        Args:
            block_duration: Duration of audio blocks to analyze (seconds)
            verbose: Whether to print progress information
        """
        self.block_duration = block_duration
        self.verbose = verbose
        self.analyzer = None
        self.sample_rate = None
        
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return samples and sample rate"""
        audio, sr = sf.read(filepath)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize to [-1, 1]
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio, sr
    
    def _analyze_with_profile(self, audio: np.ndarray, profile: CodecProfile) -> Dict[str, float]:
        """
        Analyze audio with specific codec profile
        Returns dictionary of metrics
        """
        metrics_list = []
        indices_list = []
        
        # Determine block size in samples
        block_samples = int(self.block_duration * self.sample_rate)
        
        # Analyze each block
        num_blocks = min(10, len(audio) // block_samples)  # Limit to 10 blocks for speed
        
        for block_idx in range(num_blocks):
            start = block_idx * block_samples
            end = start + block_samples
            block = audio[start:end]
            
            if len(block) < block_samples:
                continue
            
            # Choose analysis based on transform type
            if profile.transform_type == TransformType.MDCT:
                metrics = self._analyze_mdct_block(block, profile)
            elif profile.transform_type == TransformType.ACELP:
                metrics = self._analyze_acelp_block(block, profile)
            elif profile.transform_type == TransformType.LP:
                metrics = self._analyze_lp_block(block, profile)
            elif profile.transform_type == TransformType.PCM:
                metrics = self._analyze_pcm_block(block, profile)
            elif profile.transform_type == TransformType.SUBBAND:
                metrics = self._analyze_subband_block(block, profile)
            elif profile.transform_type == TransformType.HYBRID:
                metrics = self._analyze_hybrid_block(block, profile)
            else:
                metrics = self._analyze_mdct_block(block, profile)  # Default to MDCT
            
            metrics_list.append(metrics)
            indices_list.append(start)
        
        # Combine estimates from all blocks
        if len(metrics_list) > 0:
            # Use appropriate period based on codec
            period = profile.frame_sizes[0] * profile.hop_size_ratio
            final_score = self.analyzer.combine_block_estimates(
                metrics_list, indices_list, int(period)
            )
            
            # Calculate average metrics
            avg_metrics = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                avg_metrics[key] = np.mean(values)
            
            avg_metrics['final_score'] = final_score
            return avg_metrics
        
        return {'final_score': 0}
    
    def _analyze_mdct_block(self, block: np.ndarray, profile: CodecProfile) -> Dict[str, float]:
        """Analyze block using MDCT (for MP3, AAC, AC-3, Vorbis)"""
        window_size = profile.frame_sizes[0]
        hop_size = int(window_size * profile.hop_size_ratio)
        
        # Map window type
        window_type_map = {
            'sine': 'sine',
            'kbd': 'kbd',
            'slope': 'slope'
        }
        window_type = window_type_map.get(profile.window_type.value, 'sine')
        
        # Compute MDCT spectrogram
        spectrogram = self.analyzer.compute_spectrogram_mdct(
            block, window_size, hop_size, 
            window_type=window_type,
            window_params=profile.window_params
        )
        
        # Detect compression traces
        metrics = self.analyzer.detect_compression_traces(spectrogram)
        
        return metrics
    
    def _analyze_acelp_block(self, block: np.ndarray, profile: CodecProfile) -> Dict[str, float]:
        """Analyze block using ACELP (for AMR-NB/WB)"""
        frame_size = profile.frame_sizes[0]
        lp_order = profile.additional_params.get('lp_order', 10)
        
        # Compute LP residual
        residuals, lp_coeffs = self.analyzer.compute_lp_residual(
            block, frame_size, lp_order
        )
        
        metrics = {}
        
        if len(residuals) > 0:
            # Analyze residual for quantization artifacts
            residual_flat = residuals.flatten()
            
            # Check for sparse excitation (characteristic of ACELP)
            sparsity = np.sum(np.abs(residual_flat) < 0.01) / len(residual_flat)
            metrics['residual_sparsity'] = sparsity
            
            # Check for pulse patterns
            metrics['residual_variance'] = np.var(residual_flat)
            metrics['residual_kurtosis'] = self._calculate_kurtosis(residual_flat)
            
            # LP coefficient quantization artifacts
            if len(lp_coeffs) > 0:
                lp_variance = np.var(lp_coeffs.flatten())
                metrics['lp_variance'] = lp_variance
        else:
            metrics = {
                'residual_sparsity': 0,
                'residual_variance': 0,
                'residual_kurtosis': 0,
                'lp_variance': 0
            }
        
        # Calculate final score based on ACELP characteristics
        metrics['max_diff'] = metrics['residual_sparsity'] * 10  # Scale for compatibility
        
        return metrics
    
    def _analyze_lp_block(self, block: np.ndarray, profile: CodecProfile) -> Dict[str, float]:
        """Analyze block using LP (for iLBC, GSM-FR, SILK)"""
        frame_size = profile.frame_sizes[0]
        lp_order = profile.additional_params.get('lpc_order', 10)
        
        # Similar to ACELP but with different emphasis
        residuals, lp_coeffs = self.analyzer.compute_lp_residual(
            block, frame_size, lp_order
        )
        
        metrics = {}
        
        if len(residuals) > 0:
            residual_flat = residuals.flatten()
            
            # LP-based codecs have different residual characteristics
            metrics['residual_energy'] = np.mean(residual_flat ** 2)
            metrics['residual_entropy'] = self._calculate_entropy(residual_flat)
            
            # Check for RPE patterns (for GSM)
            if profile.additional_params.get('rpe', False):
                metrics['rpe_score'] = self._detect_rpe_patterns(residual_flat)
            else:
                metrics['rpe_score'] = 0
        else:
            metrics = {
                'residual_energy': 0,
                'residual_entropy': 0,
                'rpe_score': 0
            }
        
        metrics['max_diff'] = metrics['residual_entropy']  # Scale for compatibility
        
        return metrics
    
    def _analyze_pcm_block(self, block: np.ndarray, profile: CodecProfile) -> Dict[str, float]:
        """Analyze block for PCM/companding (G.711)"""
        metrics = self.analyzer.analyze_g711_artifacts(block)
        
        # Add G.711-specific checks
        # Check for 8-bit quantization patterns
        unique_values = len(np.unique(np.round(block * 32768)))
        metrics['unique_values'] = unique_values
        
        # Check for companding curve
        if unique_values <= 256:  # Likely 8-bit
            metrics['g711_probability'] = 0.9
        else:
            metrics['g711_probability'] = 0.1
        
        metrics['max_diff'] = metrics.get('quantization_score', 0) * 10
        
        return metrics
    
    def _analyze_subband_block(self, block: np.ndarray, profile: CodecProfile) -> Dict[str, float]:
        """Analyze block for sub-band coding (G.722)"""
        cutoff = profile.frequency_bands[0][1] if profile.frequency_bands else 4000
        metrics = self.analyzer.analyze_subband_artifacts(block, cutoff)
        
        # G.722-specific analysis
        metrics['qmf_score'] = self._detect_qmf_artifacts(block)
        
        metrics['max_diff'] = metrics.get('subband_energy_ratio', 0)
        
        return metrics
    
    def _analyze_hybrid_block(self, block: np.ndarray, profile: CodecProfile) -> Dict[str, float]:
        """Analyze block for hybrid coding (EVS, Opus, MP3)"""
        # Combine multiple analysis methods
        
        # Try MDCT analysis
        mdct_metrics = self._analyze_mdct_block(block, profile)
        
        # Try LP analysis
        lp_metrics = self._analyze_lp_block(block, profile)
        
        # Combine metrics
        combined_metrics = {}
        for key in mdct_metrics:
            combined_metrics[f'mdct_{key}'] = mdct_metrics[key]
        for key in lp_metrics:
            combined_metrics[f'lp_{key}'] = lp_metrics[key]
        
        # Hybrid score is maximum of both
        combined_metrics['max_diff'] = max(
            mdct_metrics.get('max_diff', 0),
            lp_metrics.get('max_diff', 0)
        )
        
        return combined_metrics
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) == 0:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, data: np.ndarray, bins: int = 50) -> float:
        """Calculate entropy of data distribution"""
        hist, _ = np.histogram(data, bins=bins)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
    
    def _detect_rpe_patterns(self, residual: np.ndarray) -> float:
        """Detect Regular Pulse Excitation patterns (GSM-FR)"""
        # GSM uses RPE with specific pulse spacing
        # Simple detection based on autocorrelation
        autocorr = np.correlate(residual, residual, mode='same')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for peaks at regular intervals
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr)
        
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            regularity = 1.0 / (1.0 + np.std(intervals))
            return regularity
        return 0
    
    def _detect_qmf_artifacts(self, block: np.ndarray) -> float:
        """Detect QMF filterbank artifacts (G.722)"""
        # QMF creates specific aliasing patterns
        # Simple detection using frequency analysis
        fft = np.fft.fft(block)
        fft_mag = np.abs(fft[:len(fft)//2])
        
        # Check for energy concentration at specific frequencies
        mid_point = len(fft_mag) // 2
        low_energy = np.sum(fft_mag[:mid_point])
        high_energy = np.sum(fft_mag[mid_point:])
        
        ratio = high_energy / (low_energy + 1e-10)
        
        # QMF typically creates a specific ratio
        target_ratio = 0.2  # Empirical value for G.722
        score = 1.0 / (1.0 + abs(ratio - target_ratio))
        
        return score
    
    def identify(self, wav_filepath: str) -> Dict[str, float]:
        """
        Main method to identify codec from WAV file
        
        Args:
            wav_filepath: Path to WAV file
            
        Returns:
            Dictionary with codec names as keys and probabilities as values
        """
        # Load audio
        audio, sr = self.load_audio(wav_filepath)
        self.sample_rate = sr
        self.analyzer = CompressionAnalyzer(sr)
        
        if self.verbose:
            print(f"Loaded audio: {len(audio)/sr:.2f} seconds at {sr} Hz")
        
        # Analyze with each codec profile
        scores = {}
        
        codec_iterator = tqdm(CODEC_PROFILES.items()) if self.verbose else CODEC_PROFILES.items()
        
        for codec_name, profile in codec_iterator:
            if self.verbose:
                codec_iterator.set_description(f"Analyzing {codec_name}")
            
            # Check if sample rate matches
            if sr in profile.sample_rates:
                metrics = self._analyze_with_profile(audio, profile)
                scores[codec_name] = metrics.get('final_score', 0)
            else:
                scores[codec_name] = 0  # Codec doesn't support this sample rate
        
        # Convert scores to probabilities using softmax
        probabilities = self._scores_to_probabilities(scores)
        
        return probabilities
    
    def _scores_to_probabilities(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert raw scores to probabilities using softmax"""
        # Get scores as array
        codec_names = list(scores.keys())
        score_values = np.array([scores[name] for name in codec_names])
        
        # Apply softmax with temperature
        temperature = 0.5  # Lower temperature makes distribution more peaked
        
        # Prevent overflow in exp
        score_values = score_values - np.max(score_values)
        exp_scores = np.exp(score_values / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Create probability dictionary
        prob_dict = {}
        for name, prob in zip(codec_names, probabilities):
            prob_dict[name] = float(prob)
        
        # Sort by probability
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        
        return prob_dict
    
    def identify_with_details(self, wav_filepath: str) -> Dict:
        """
        Identify codec with detailed analysis results
        
        Returns dictionary with probabilities and detailed metrics
        """
        probabilities = self.identify(wav_filepath)
        
        # Get top 3 most likely codecs
        top_codecs = list(probabilities.keys())[:3]
        
        result = {
            'file': wav_filepath,
            'probabilities': probabilities,
            'top_codec': top_codecs[0],
            'confidence': probabilities[top_codecs[0]],
            'top_3': [
                {'codec': codec, 'probability': probabilities[codec]}
                for codec in top_codecs
            ]
        }
        
        return result