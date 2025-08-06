#!/usr/bin/env python3
"""
Comprehensive Test Runner for Synthformer
=========================================

This script runs all tests for the Synthformer project including:
- Model architecture tests
- Training pipeline tests  
- Dataloader tests
- Utils function tests
- Integration tests

Usage:
    python test_runner.py [--module MODULE] [--verbose] [--coverage] [--quick]

Options:
    --module MODULE    Run tests for specific module only (model, training, dataloader, utils, integration)
    --verbose         Verbose output
    --coverage        Generate coverage report
    --quick           Run only fast tests (skip slow integration tests)
    --help           Show this help message
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path


class TestRunner:
    """Comprehensive test runner for Synthformer"""
    
    def __init__(self, verbose=False, coverage=False, quick=False):
        self.verbose = verbose
        self.coverage = coverage
        self.quick = quick
        self.test_results = {}
        self.start_time = time.time()
        
        # Define test modules
        self.test_modules = {
            'model': 'test_model.py',
            'training': 'test_training.py', 
            'dataloader': 'test_dataloader_extended.py',
            'utils': 'test_utils.py',
            'integration': 'test_integration.py',
            'original': 'test_Dataloader.py'  # Original test
        }
    
    def print_header(self):
        """Print test runner header"""
        print("=" * 80)
        print("🧪 SYNTHFORMER COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Starting test run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print()
    
    def print_footer(self):
        """Print test runner footer with summary"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        total_passed = sum(1 for result in self.test_results.values() if result['passed'])
        total_failed = sum(1 for result in self.test_results.values() if not result['passed'])
        total_time = time.time() - self.start_time
        
        print(f"Total modules tested: {len(self.test_results)}")
        print(f"Modules passed: {total_passed}")
        print(f"Modules failed: {total_failed}")
        print(f"Total test time: {total_time:.2f} seconds")
        
        if total_failed > 0:
            print("\n❌ FAILED MODULES:")
            for module, result in self.test_results.items():
                if not result['passed']:
                    print(f"  - {module}: {result.get('error', 'Unknown error')}")
        
        if total_failed == 0:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {total_failed} MODULE(S) FAILED")
        
        print("=" * 80)
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        print("🔍 Checking dependencies...")
        
        required_modules = [
            'torch', 'numpy', 'pytest', 'rdkit', 'sklearn', 'pandas'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                if self.verbose:
                    print(f"  ✅ {module}")
            except ImportError:
                missing_modules.append(module)
                print(f"  ❌ {module}")
        
        if missing_modules:
            print(f"\n⚠️  Missing required modules: {', '.join(missing_modules)}")
            print("Please install missing dependencies before running tests.")
            return False
        
        print("✅ All dependencies available\n")
        return True
    
    def run_pytest_module(self, module_name, test_file):
        """Run tests for a specific module using pytest"""
        print(f"🧪 Running {module_name} tests ({test_file})...")
        
        if not os.path.exists(test_file):
            print(f"  ⚠️  Test file {test_file} not found, skipping...")
            self.test_results[module_name] = {
                'passed': False,
                'error': f'Test file {test_file} not found'
            }
            return False
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest', test_file]
        
        if self.verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.append('-q')
        
        if self.quick and 'integration' in module_name:
            cmd.extend(['-m', 'not slow'])
        
        if self.coverage:
            cmd.extend(['--cov=.', '--cov-report=term-missing'])
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                timeout=300  # 5 minute timeout per module
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"  ✅ {module_name} tests PASSED ({duration:.2f}s)")
                self.test_results[module_name] = {
                    'passed': True,
                    'duration': duration
                }
                return True
            else:
                error_msg = result.stderr if result.stderr else "Tests failed"
                print(f"  ❌ {module_name} tests FAILED ({duration:.2f}s)")
                if self.verbose and result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                    
                self.test_results[module_name] = {
                    'passed': False,
                    'duration': duration,
                    'error': error_msg
                }
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {module_name} tests TIMED OUT")
            self.test_results[module_name] = {
                'passed': False,
                'error': 'Test timed out after 5 minutes'
            }
            return False
            
        except Exception as e:
            print(f"  💥 {module_name} tests CRASHED: {str(e)}")
            self.test_results[module_name] = {
                'passed': False,
                'error': str(e)
            }
            return False
    
    def run_specific_module(self, module_name):
        """Run tests for a specific module"""
        if module_name not in self.test_modules:
            print(f"❌ Unknown module: {module_name}")
            print(f"Available modules: {', '.join(self.test_modules.keys())}")
            return False
        
        test_file = self.test_modules[module_name]
        return self.run_pytest_module(module_name, test_file)
    
    def run_all_tests(self):
        """Run all test modules"""
        print("🚀 Running all test modules...\n")
        
        # Define test order (dependencies first)
        test_order = ['utils', 'model', 'dataloader', 'training', 'original', 'integration']
        
        all_passed = True
        
        for module_name in test_order:
            if module_name in self.test_modules:
                success = self.run_pytest_module(module_name, self.test_modules[module_name])
                all_passed = all_passed and success
                print()  # Add spacing between modules
        
        return all_passed
    
    def generate_coverage_report(self):
        """Generate detailed coverage report"""
        if not self.coverage:
            return
        
        print("📈 Generating coverage report...")
        
        try:
            # Generate HTML coverage report
            subprocess.run([
                'python', '-m', 'pytest', 
                '--cov=.', 
                '--cov-report=html',
                '--cov-report=term'
            ] + list(self.test_modules.values()), 
            check=True)
            
            print("✅ Coverage report generated in htmlcov/")
            
        except subprocess.CalledProcessError:
            print("⚠️  Failed to generate coverage report")
    
    def run_linting(self):
        """Run code linting checks"""
        print("🧹 Running code quality checks...")
        
        # Check if flake8 is available
        try:
            subprocess.run(['flake8', '--version'], 
                         capture_output=True, check=True)
            
            # Run flake8 on Python files
            result = subprocess.run([
                'flake8', 
                '--max-line-length=100',
                '--ignore=E501,W503',  # Ignore line length and line break before binary operator
                '*.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Code quality checks passed")
            else:
                print("⚠️  Code quality issues found:")
                print(result.stdout)
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  flake8 not available, skipping code quality checks")
    
    def check_test_completeness(self):
        """Check that all main modules have corresponding tests"""
        print("🔍 Checking test completeness...")
        
        main_modules = ['model.py', 'train.py', 'Dataloader.py', 'utils.py']
        test_files = list(self.test_modules.values())
        
        missing_tests = []
        for module in main_modules:
            module_name = module.replace('.py', '')
            has_test = any(module_name in test_file for test_file in test_files)
            
            if has_test:
                print(f"  ✅ {module}")
            else:
                print(f"  ❌ {module} (no corresponding test)")
                missing_tests.append(module)
        
        if missing_tests:
            print(f"⚠️  Missing tests for: {', '.join(missing_tests)}")
        else:
            print("✅ All main modules have tests")
        
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Synthformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--module', 
        choices=list(TestRunner({}).test_modules.keys()),
        help='Run tests for specific module only'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true', 
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run only fast tests (skip slow integration tests)'
    )
    
    parser.add_argument(
        '--lint',
        action='store_true',
        help='Run code quality checks'
    )
    
    parser.add_argument(
        '--check-completeness',
        action='store_true',
        help='Check test completeness'
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(
        verbose=args.verbose,
        coverage=args.coverage,
        quick=args.quick
    )
    
    # Print header
    runner.print_header()
    
    # Check dependencies
    if not runner.check_dependencies():
        return 1
    
    # Check test completeness if requested
    if args.check_completeness:
        runner.check_test_completeness()
    
    # Run linting if requested
    if args.lint:
        runner.run_linting()
        print()
    
    # Run tests
    if args.module:
        success = runner.run_specific_module(args.module)
    else:
        success = runner.run_all_tests()
    
    # Generate coverage report if requested
    if args.coverage:
        runner.generate_coverage_report()
    
    # Print footer
    runner.print_footer()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 