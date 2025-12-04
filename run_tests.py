#!/usr/bin/env python
"""
Test runner script for the project
Run all tests or specific test suites
"""

import sys
import subprocess
import argparse


def run_tests(test_path=None, verbose=False, coverage=False, markers=None):
    """
    Run tests using pytest
    
    Args:
        test_path: Specific test file or directory (None = all tests)
        verbose: Run with verbose output
        coverage: Generate coverage report
        markers: Run tests with specific markers
    """
    cmd = ['pytest']
    
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append('tests/')
    
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend([
            '--cov=src',
            '--cov-report=html',
            '--cov-report=term-missing'
        ])
    
    if markers:
        cmd.extend(['-m', markers])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run unit tests for the project'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Run specific test file (e.g., test_models.py)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '-m', '--markers',
        type=str,
        help='Run tests with specific markers (e.g., "unit", "integration")'
    )
    parser.add_argument(
        '--models',
        action='store_true',
        help='Run model tests only'
    )
    parser.add_argument(
        '--data',
        action='store_true',
        help='Run data loader tests only'
    )
    parser.add_argument(
        '--tracking',
        action='store_true',
        help='Run tracking tests only'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run integration tests only'
    )
    
    args = parser.parse_args()
    
    # Determine test path
    test_path = None
    if args.test:
        test_path = f'tests/{args.test}'
    elif args.models:
        test_path = 'tests/test_models.py'
    elif args.data:
        test_path = 'tests/test_data_loader.py'
    elif args.tracking:
        test_path = 'tests/test_tracking.py'
    elif args.integration:
        test_path = 'tests/test_integration.py'
    
    # Run tests
    exit_code = run_tests(
        test_path=test_path,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=args.markers
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()


