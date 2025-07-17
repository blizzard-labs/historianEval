#!/usr/bin/env python3
"""
HISTORIAN Log Parser - Calculate ESS Values
Parses HISTORIAN MCMC log files to calculate scalar and topological ESS values.

This script implements sophisticated autocorrelation analysis using:
1. FFT-based autocorrelation computation for efficiency
2. Automatic windowing (Sokal method) for robust τ_int estimation
3. Detailed diagnostic information for MCMC assessment
"""

import re
import numpy as np
from collections import defaultdict
import argparse
import os
import matplotlib.pyplot as plt

class HistorianLogParser:
    def __init__(self, log_file):
        self.log_file = log_file
        self.scalar_params = {}
        self.topological_moves = []
        self.move_stats = defaultdict(lambda: {'moves': 0, 'accepted': 0, 'time': 0.0})
        
    def parse_log_file(self):
        """Parse the HISTORIAN log file and extract parameters and move statistics."""
        print(f"Parsing log file: {self.log_file}")
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        # Parse scalar parameters (log-likelihood values, etc.)
        self._parse_scalar_parameters(lines)
        
        # Parse move statistics
        self._parse_move_statistics(lines)
        
        print(f"Found {len(self.scalar_params)} scalar parameters")
        print(f"Found {len(self.move_stats)} move types")
        
    def _parse_scalar_parameters(self, lines):
        """Extract scalar parameters like log-likelihood values."""
        param_patterns = [
            r'log-likelihood\s+is\s+([-\d.]+)',
            r'profile\s+log-likelihood\s+is\s+([-\d.]+)',
            r'final\s+alignment\s+log-likelihood\s+is\s+([-\d.]+)',
            r'New\s+best\s+log-likelihood:\s+([-\d.]+)'
        ]
        
        for line in lines:
            for pattern in param_patterns:
                match = re.search(pattern, line)
                if match:
                    value = float(match.group(1))
                    param_name = pattern.split(r'\s+')[0].replace('\\', '')
                    
                    if param_name not in self.scalar_params:
                        self.scalar_params[param_name] = []
                    self.scalar_params[param_name].append(value)
    
    def _parse_move_statistics(self, lines):
        """Parse move statistics from the summary sections."""
        in_summary = False
        
        for line in lines:
            line = line.strip()
            
            # Check for dataset summary section
            if line.startswith('Dataset #'):
                in_summary = True
                continue
            
            if in_summary and ':' in line:
                # Parse move statistics like "Branch alignment: 272 moves, 204 accepted, 8.00937 seconds, 25.4702 accepted/sec"
                parts = line.split(':')
                if len(parts) == 2:
                    move_type = parts[0].strip()
                    stats_str = parts[1].strip()
                    
                    # Extract numbers using regex
                    numbers = re.findall(r'(\d+(?:\.\d+)?)', stats_str)
                    if len(numbers) >= 4:
                        moves = int(float(numbers[0]))
                        accepted = int(float(numbers[1]))
                        time_sec = float(numbers[2])
                        
                        self.move_stats[move_type] = {
                            'moves': moves,
                            'accepted': accepted,
                            'time': time_sec,
                            'acceptance_rate': accepted / moves if moves > 0 else 0
                        }
    
    def calculate_scalar_ess(self, burnin_fraction=0.1):
        """Calculate ESS for scalar parameters using autocorrelation."""
        ess_results = {}
        
        for param_name, values in self.scalar_params.items():
            if len(values) < 10:  # Need sufficient samples
                continue
                
            # Remove burnin
            burnin_samples = int(len(values) * burnin_fraction)
            post_burnin = values[burnin_samples:]
            
            if len(post_burnin) < 5:
                continue
            
            # Calculate ESS using autocorrelation
            ess_data = self._calculate_ess_autocorr(post_burnin)
            ess_results[param_name] = {
                'ess': ess_data['ess'],
                'tau_int': ess_data['tau_int'],
                'n_samples': len(post_burnin),
                'mean': np.mean(post_burnin),
                'std': np.std(post_burnin),
                'autocorr': ess_data['autocorr'],
                'window_size': ess_data['window_size']
            }
        
        return ess_results
    
    def calculate_topological_ess(self):
        """Calculate topological ESS based on tree-changing moves."""
        topological_moves = [
            'Branch alignment',
            'Node alignment', 
            'Prune-and-regraft',
            'Node height',
            'Rescale'
        ]
        
        total_tree_moves = 0
        total_accepted_tree_moves = 0
        
        for move_type in topological_moves:
            if move_type in self.move_stats:
                stats = self.move_stats[move_type]
                total_tree_moves += stats['moves']
                total_accepted_tree_moves += stats['accepted']
        
        if total_tree_moves == 0:
            return {'topological_ess': 0, 'tree_acceptance_rate': 0}
        
        # Estimate topological ESS based on acceptance rates
        # This is a simplified approximation - more sophisticated methods exist
        tree_acceptance_rate = total_accepted_tree_moves / total_tree_moves
        
        # ESS approximation: accepted moves adjusted for autocorrelation
        # Rule of thumb: ESS ≈ accepted_moves / (2 * autocorr_time)
        # For tree topology, autocorr_time is typically higher
        autocorr_factor = 2.0  # Conservative estimate
        topological_ess = total_accepted_tree_moves / autocorr_factor
        
        return {
            'topological_ess': topological_ess,
            'tree_acceptance_rate': tree_acceptance_rate,
            'total_tree_moves': total_tree_moves,
            'total_accepted_tree_moves': total_accepted_tree_moves
        }
    
    def _calculate_ess_autocorr(self, x):
        """Calculate ESS using autocorrelation method with automatic windowing."""
        x = np.array(x)
        n = len(x)
        
        if n < 4:
            return {'ess': n, 'tau_int': 0, 'autocorr': [1.0]}
        
        # Center the data
        x_centered = x - np.mean(x)
        
        # Calculate autocorrelation using FFT for efficiency
        # Pad with zeros to avoid circular correlation
        padded = np.zeros(2 * n)
        padded[:n] = x_centered
        
        # FFT-based autocorrelation
        fft_result = np.fft.fft(padded)
        autocorr_fft = np.fft.ifft(fft_result * np.conj(fft_result)).real
        autocorr = autocorr_fft[:n]
        autocorr = autocorr / autocorr[0]  # Normalize by variance
        
        # Automatic windowing (Sokal method)
        c = 5  # Window factor
        W = 1
        tau_int = 1.0
        
        max_window = min(n//4, 200)  # Reasonable maximum window
        
        while W < max_window:
            if W < len(autocorr):
                tau_int = 1 + 2 * np.sum(autocorr[1:W+1])
                
                # Check if window is large enough (W >= c * tau_int)
                if W >= c * tau_int:
                    break
            W += 1
        
        # Calculate ESS
        ess = n / (2 * tau_int + 1)
        
        return {
            'ess': max(1, ess),
            'tau_int': tau_int,
            'autocorr': autocorr[:min(100, len(autocorr))],  # Keep first 100 lags
            'window_size': W
        }
    
    def print_results(self):
        """Print comprehensive ESS analysis results."""
        print("\n" + "="*60)
        print("HISTORIAN ESS ANALYSIS RESULTS")
        print("="*60)
        
        # Move statistics
        print("\nMove Statistics:")
        print("-" * 50)
        for move_type, stats in self.move_stats.items():
            print(f"{move_type}:")
            print(f"  Moves: {stats['moves']}, Accepted: {stats['accepted']}")
            print(f"  Acceptance Rate: {stats['acceptance_rate']:.3f}")
            print(f"  Time: {stats['time']:.3f} seconds")
            print()
        
        # Scalar ESS
        scalar_ess = self.calculate_scalar_ess()
        if scalar_ess:
            print("Scalar Parameter ESS:")
            print("-" * 50)
            for param, results in scalar_ess.items():
                print(f"{param}:")
                print(f"  ESS: {results['ess']:.1f}")
                print(f"  Integrated Autocorr Time (τ): {results['tau_int']:.2f}")
                print(f"  Window Size: {results['window_size']}")
                print(f"  Samples: {results['n_samples']}")
                print(f"  Mean: {results['mean']:.3f}")
                print(f"  Std: {results['std']:.3f}")
                
                # Autocorrelation decay assessment
                autocorr = results['autocorr']
                if len(autocorr) > 10:
                    decay_10 = autocorr[10] if len(autocorr) > 10 else 0
                    print(f"  Autocorr at lag 10: {decay_10:.3f}")
                print()
        
        # Topological ESS
        topo_ess = self.calculate_topological_ess()
        print("Topological ESS:")
        print("-" * 50)
        print(f"Topological ESS: {topo_ess['topological_ess']:.1f}")
        print(f"Tree Acceptance Rate: {topo_ess['tree_acceptance_rate']:.3f}")
        print(f"Total Tree Moves: {topo_ess['total_tree_moves']}")
        print(f"Total Accepted Tree Moves: {topo_ess['total_accepted_tree_moves']}")
        
        # ESS recommendations
        print("\nESS Recommendations:")
        print("-" * 50)
        min_ess = 200  # Common threshold
        
        if scalar_ess:
            for param, results in scalar_ess.items():
                if results['ess'] < min_ess:
                    print(f"⚠️  {param}: ESS ({results['ess']:.1f}) < {min_ess} - Consider longer run")
                else:
                    print(f"✅ {param}: ESS ({results['ess']:.1f}) sufficient")
        
        if topo_ess['topological_ess'] < min_ess:
            print(f"⚠️  Topology: ESS ({topo_ess['topological_ess']:.1f}) < {min_ess} - Consider longer run")
    def plot_autocorrelation(self, param_name, max_lag=50):
        """Plot autocorrelation function for a specific parameter."""
        scalar_ess = self.calculate_scalar_ess()
        
        if param_name not in scalar_ess:
            print(f"Parameter '{param_name}' not found in results")
            return
        
        results = scalar_ess[param_name]
        autocorr = results['autocorr']
        tau_int = results['tau_int']
        
        plt.figure(figsize=(10, 6))
        lags = np.arange(min(len(autocorr), max_lag))
        plt.plot(lags, autocorr[:len(lags)], 'b-', linewidth=2, label='Autocorrelation')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=np.exp(-1), color='r', linestyle='--', alpha=0.7, 
                   label=f'1/e ≈ 0.368')
        
        # Mark integrated autocorrelation time
        if tau_int < max_lag:
            plt.axvline(x=tau_int, color='g', linestyle='--', alpha=0.7,
                       label=f'τ_int = {tau_int:.2f}')
        
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title(f'Autocorrelation Function: {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_effective_sample_size_summary(self):
        """Get a summary of ESS values for reporting."""
        scalar_ess = self.calculate_scalar_ess()
        topo_ess = self.calculate_topological_ess()
        
        summary = {
            'scalar_parameters': {},
            'topological': topo_ess,
            'overall_assessment': 'good'
        }
        
        min_ess = 200
        problematic_params = []
        
        for param, results in scalar_ess.items():
            summary['scalar_parameters'][param] = {
                'ess': results['ess'],
                'tau_int': results['tau_int'],
                'sufficient': results['ess'] >= min_ess
            }
            
            if results['ess'] < min_ess:
                problematic_params.append(param)
        
        if problematic_params or topo_ess['topological_ess'] < min_ess:
            summary['overall_assessment'] = 'problematic'
            summary['recommendations'] = []
            
            if problematic_params:
                summary['recommendations'].append(
                    f"Increase chain length for parameters: {', '.join(problematic_params)}"
                )
            
            if topo_ess['topological_ess'] < min_ess:
                summary['recommendations'].append(
                    "Increase chain length for better topological sampling"
                )
        
        else:
            print(f"✅ Topology: ESS ({topo_ess['topological_ess']:.1f}) sufficient")
        
        print(f"\nOverall Assessment: {self.get_effective_sample_size_summary()['overall_assessment'].upper()}")
        return summary

def main():
    parser = argparse.ArgumentParser(description='Parse HISTORIAN log files and calculate ESS values')
    parser.add_argument('log_file', help='Path to HISTORIAN log file')
    parser.add_argument('--burnin', type=float, default=0.1, 
                       help='Burnin fraction (default: 0.1)')
    parser.add_argument('--plot', type=str, default="yes",
                       help='Plot autocorrelation for specific parameter')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found")
        return
    
    # Parse the log file
    log_parser = HistorianLogParser(args.log_file)
    log_parser.parse_log_file()
    log_parser.print_results()
    
    # Optional plotting
    if args.plot:
        try:
            log_parser.plot_autocorrelation(args.plot)
        except ImportError:
            print("Warning: matplotlib not available for plotting")
        except Exception as e:
            print(f"Error plotting autocorrelation: {e}")

if __name__ == "__main__":
    main()