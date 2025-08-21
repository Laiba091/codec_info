#!/usr/bin/env python3
"""
Audio Codec Identifier - Main CLI Interface
Identifies the source codec of a WAV file based on compression artifacts
"""

import argparse
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.codec_identifier import CodecIdentifier
from src.codec_profiles import get_all_codec_names

def print_results(results: dict, format: str = 'pretty'):
    """Print results in specified format"""
    
    if format == 'json':
        print(json.dumps(results, indent=2))
    
    elif format == 'csv':
        print("Codec,Probability")
        for codec, prob in results['probabilities'].items():
            print(f"{codec},{prob:.6f}")
    
    else:  # pretty
        print("\n" + "="*60)
        print(f"Audio Codec Identification Results")
        print("="*60)
        print(f"File: {results['file']}")
        print(f"\nMost Likely Codec: {results['top_codec']}")
        print(f"Confidence: {results['confidence']:.2%}")
        
        print(f"\nTop 3 Predictions:")
        print("-"*40)
        for i, pred in enumerate(results['top_3'], 1):
            print(f"{i}. {pred['codec']:12s} - {pred['probability']:.2%}")
        
        print(f"\nAll Probabilities:")
        print("-"*40)
        for codec, prob in results['probabilities'].items():
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"{codec:12s} [{bar}] {prob:.4f}")
        
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Identify audio codec from WAV file based on compression artifacts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav                    # Basic identification
  %(prog)s audio.wav -v                 # Verbose mode
  %(prog)s audio.wav -f json            # JSON output
  %(prog)s audio.wav -o results.json    # Save to file
  
Supported Codecs:
  Music/General: MP3, AAC, AC-3, Vorbis
  Network/VoIP:  G.711, G.722, AMR-NB, AMR-WB, EVS, Opus
  Legacy:        GSM-FR, iLBC, Speex, SILK
        """
    )
    
    parser.add_argument('input', 
                       nargs='?',
                       help='Path to WAV file to analyze')
    
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('-f', '--format', 
                       choices=['pretty', 'json', 'csv'],
                       default='pretty',
                       help='Output format (default: pretty)')
    
    parser.add_argument('-o', '--output', 
                       help='Save results to file')
    
    parser.add_argument('-b', '--block-duration', 
                       type=float, 
                       default=1.0,
                       help='Duration of audio blocks to analyze in seconds (default: 1.0)')
    
    parser.add_argument('--list-codecs', 
                       action='store_true',
                       help='List all supported codecs and exit')
    
    args = parser.parse_args()
    
    # List codecs if requested
    if args.list_codecs:
        print("\nSupported Codecs:")
        print("-" * 40)
        for codec in get_all_codec_names():
            print(f"  {codec}")
        print()
        return 0
    
    # Check if input file was provided
    if not args.input:
        parser.error("Input file is required unless using --list-codecs")
        return 1
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        return 1
    
    if not input_path.suffix.lower() in ['.wav', '.wave']:
        print(f"Warning: File does not have .wav extension", file=sys.stderr)
    
    try:
        # Initialize identifier
        identifier = CodecIdentifier(
            block_duration=args.block_duration,
            verbose=args.verbose
        )
        
        # Run identification
        if args.verbose:
            print(f"\nAnalyzing: {args.input}")
            print("-" * 40)
        
        results = identifier.identify_with_details(str(input_path))
        
        # Output results
        if args.output:
            # Save to file
            output_path = Path(args.output)
            
            if args.format == 'json' or output_path.suffix == '.json':
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            elif args.format == 'csv' or output_path.suffix == '.csv':
                with open(output_path, 'w') as f:
                    f.write("Codec,Probability\n")
                    for codec, prob in results['probabilities'].items():
                        f.write(f"{codec},{prob:.6f}\n")
            else:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            print(f"Results saved to: {output_path}")
        else:
            # Print to console
            print_results(results, args.format)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())