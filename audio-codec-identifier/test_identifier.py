#!/usr/bin/env python3
"""
Test script for codec identifier
Creates test audio files and verifies identification
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from src.codec_identifier import CodecIdentifier

def create_test_audio(duration=2.0, sample_rate=16000):
    """Create a test audio signal"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a signal with multiple frequency components
    signal = np.zeros_like(t)
    
    # Add some tones
    frequencies = [440, 880, 1320]  # A4, A5, E6
    for freq in frequencies:
        signal += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    signal += 0.05 * np.random.randn(len(t))
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return signal, sample_rate

def test_basic_identification():
    """Test basic codec identification"""
    print("Testing Basic Codec Identification")
    print("-" * 40)
    
    # Create test audio
    audio, sr = create_test_audio()
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name
    
    try:
        # Initialize identifier
        identifier = CodecIdentifier(verbose=True)
        
        # Run identification
        results = identifier.identify_with_details(tmp_path)
        
        # Print results
        print(f"\nTop identified codec: {results['top_codec']}")
        print(f"Confidence: {results['confidence']:.2%}")
        
        print("\nTop 3 predictions:")
        for pred in results['top_3']:
            print(f"  {pred['codec']:12s} - {pred['probability']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
        
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)

def test_different_sample_rates():
    """Test with different sample rates"""
    print("\nTesting Different Sample Rates")
    print("-" * 40)
    
    sample_rates = [8000, 16000, 44100, 48000]
    identifier = CodecIdentifier(verbose=False)
    
    for sr in sample_rates:
        audio, _ = create_test_audio(duration=1.0, sample_rate=sr)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            tmp_path = tmp.name
        
        try:
            results = identifier.identify(tmp_path)
            top_codec = max(results, key=results.get)
            print(f"  {sr:5d} Hz -> {top_codec} ({results[top_codec]:.2%})")
        except Exception as e:
            print(f"  {sr:5d} Hz -> Error: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    return True

def test_performance():
    """Test identification performance"""
    print("\nTesting Performance")
    print("-" * 40)
    
    import time
    
    audio, sr = create_test_audio(duration=10.0)  # 10 second audio
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name
    
    try:
        identifier = CodecIdentifier(verbose=False)
        
        start_time = time.time()
        results = identifier.identify(tmp_path)
        elapsed = time.time() - start_time
        
        print(f"  Processed 10 seconds of audio in {elapsed:.2f} seconds")
        print(f"  Speed: {10/elapsed:.1f}x realtime")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Audio Codec Identifier - Test Suite")
    print("="*60)
    
    tests = [
        test_basic_identification,
        test_different_sample_rates,
        test_performance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())