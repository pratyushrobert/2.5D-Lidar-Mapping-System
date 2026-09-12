"""
main.py
=======
Entry point for the Adaptive Variable-Resolution 2.5D LiDAR Mapping System.

Boots directly into synthetic demo mode by default.
Optionally accepts command-line arguments for configuration.
"""

import sys
import argparse
import numpy as np

from data_loader import SyntheticLiDARGenerator, LiDARFrame, SemanticKITTILoader
from grid_engine import AdaptiveGridEngine, UniformGridEngine
from perception import GeometricPerception, PointNetPlusStub
from benchmark import BenchmarkEngine, BenchmarkResult


def run_headless_demo(n_frames: int = 50, max_range: float = 100.0):
    """
    Run a headless benchmarking demo without GUI.
    Useful for performance testing and validation.
    """
    print("=" * 60)
    print("Adaptive 2.5D LiDAR Mapping - Headless Demo Mode")
    print("=" * 60)
    
    generator = SyntheticLiDARGenerator(seed=42)
    perception = GeometricPerception()
    benchmark = BenchmarkEngine(max_range=max_range)
    
    frames = generator.generate_sequence(n_frames)
    
    print(f"\nGenerating {n_frames} synthetic LiDAR frames...")
    print(f"Grid extent: {max_range}m")
    print(f"Zone resolutions: 0.05m (0-10m), 0.10m (10-30m), "
          f"0.25m (30-60m), 0.50m (60-100m)")
    print("-" * 60)
    
    for i, frame in enumerate(frames):
        result = benchmark.process_frame(frame)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\nFrame {i + 1:3d} | Points: {result.n_points:6d} | "
                  f"Adaptive Cells: {result.adaptive_active_cells:6d} | "
                  f"Uniform Cells: {result.uniform_active_cells:8d} | "
                  f"Memory Saved: {result.memory_saved_pct:5.1f}% | "
                  f"FPS: {result.fps_adaptive:5.1f}")
    
    summary = benchmark.get_summary_stats()
    
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    print(f"Total Frames Processed:  {summary.get('total_frames', 0)}")
    print(f"Adaptive Grid Memory:    {summary.get('adaptive_avg_memory_mb', 0):.4f} MB")
    print(f"Uniform Grid Memory:     {summary.get('uniform_avg_memory_mb', 0):.4f} MB")
    print(f"Memory Saved:            {summary.get('avg_memory_saved_pct', 0):.2f}%")
    print(f"Active Cells (Adaptive): {summary.get('adaptive_active_cells', 0):,}")
    print(f"Active Cells (Uniform):  {summary.get('uniform_active_cells', 0):,}")
    print(f"Compression Ratio:       {summary.get('compression_ratio', 0):.2f}x")
    print(f"Avg Latency (Adaptive):  {summary.get('avg_adaptive_latency_ms', 0):.2f} ms")
    print(f"Avg Latency (Uniform):   {summary.get('avg_uniform_latency_ms', 0):.2f} ms")
    print(f"FPS (Adaptive):          {summary.get('fps_adaptive', 0):.1f}")
    print(f"FPS (Uniform):           {summary.get('fps_uniform', 0):.1f}")
    print("=" * 60)
    
    return benchmark


def run_gui_demo():
    """Launch the interactive GUI dashboard."""
    try:
        from app import run_dashboard
    except ImportError as e:
        print(f"Error: GUI dependencies not available ({e})")
        print("Install with: pip install -r requirements.txt")
        print("Or run in headless mode: python main.py --mode headless")
        sys.exit(1)
    print("Launching Adaptive 2.5D LiDAR Mapping Dashboard...")
    print("Controls:")
    print("  - Play/Pause: Start/stop frame playback")
    print("  - Frame Slider: Jump to specific frame")
    print("  - Speed Slider: Adjust playback speed (1x-10x)")
    print("  - Settings: Toggle PointNet++ inference")
    print()
    run_dashboard()


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Variable-Resolution 2.5D LiDAR Mapping System"
    )
    parser.add_argument(
        "--mode", choices=["gui", "headless"], default="gui",
        help="Run mode: 'gui' for interactive dashboard, 'headless' for benchmarking"
    )
    parser.add_argument(
        "--frames", type=int, default=200,
        help="Number of frames to generate (headless mode)"
    )
    parser.add_argument(
        "--range", type=float, default=100.0,
        help="Maximum grid range in meters"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to SemanticKITTI dataset (optional)"
    )
    
    args = parser.parse_args()
    
    if args.data_path:
        print(f"Loading SemanticKITTI from: {args.data_path}")
        loader = SemanticKITTILoader(args.data_path)
        n_frames = loader.count_frames()
        print(f"Found {n_frames} frames in dataset")
    
    if args.mode == "gui":
        run_gui_demo()
    else:
        run_headless_demo(n_frames=args.frames, max_range=args.range)


if __name__ == "__main__":
    main()
