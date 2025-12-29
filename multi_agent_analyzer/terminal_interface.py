"""
Terminal interface for the multi-agent data analyzer system.

Provides command-line interaction, progress display, and report presentation.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .main_controller import MainController


class TerminalInterface:
    """
    Terminal-based user interface for the multi-agent analyzer.
    
    Provides:
    - Command-line argument parsing
    - File loading and validation
    - Real-time progress updates
    - Report presentation in terminal
    """
    
    def __init__(self):
        """Initialize the terminal interface."""
        self.controller: Optional[MainController] = None
        self.gemini_api_key: Optional[str] = None
        
        # Terminal formatting
        self.width = 80
        self.separator = "=" * self.width
        self.subseparator = "-" * self.width
    
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="Multi-Agent Data Analyzer - Automated CSV analysis with AI agents",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python -m multi_agent_analyzer data.csv
  python -m multi_agent_analyzer data.csv --dict dictionary.json
  python -m multi_agent_analyzer data.csv --output report.txt --format text
  python -m multi_agent_analyzer data.csv --verbose
            """
        )
        
        # Required arguments
        parser.add_argument(
            "csv_file",
            type=str,
            help="Path to CSV file to analyze"
        )
        
        # Optional arguments
        parser.add_argument(
            "--dict",
            "--dictionary",
            dest="data_dict",
            type=str,
            help="Path to data dictionary JSON file"
        )
        
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            help="Output file path for report (default: display in terminal)"
        )
        
        parser.add_argument(
            "--format",
            "-f",
            choices=["text", "markdown", "json", "html"],
            default="text",
            help="Report output format (default: text)"
        )
        
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        parser.add_argument(
            "--api-key",
            type=str,
            help="Gemini API key (overrides .env file)"
        )
        
        parser.add_argument(
            "--no-llm",
            action="store_true",
            help="Disable LLM features (basic analysis only)"
        )
        
        return parser.parse_args()
    
    def setup_logging(self, verbose: bool = False) -> None:
        """
        Set up logging configuration.
        
        Args:
            verbose: Enable verbose logging
        """
        log_level = logging.DEBUG if verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Reduce noise from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(logging.WARNING)
    
    def print_header(self) -> None:
        """Print application header."""
        print(self.separator)
        print("Multi-Agent Data Analyzer".center(self.width))
        print("Automated CSV Analysis with AI Agents".center(self.width))
        print(self.separator)
        print()
    
    def print_section(self, title: str) -> None:
        """Print section header."""
        print(f"\n{self.subseparator}")
        print(f"{title}")
        print(self.subseparator)
    
    def validate_files(self, args: argparse.Namespace) -> bool:
        """
        Validate input files exist and are readable.
        
        Args:
            args: Parsed command-line arguments
        
        Returns:
            bool: True if all files are valid
        """
        # Check CSV file
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"❌ Error: CSV file not found: {args.csv_file}")
            return False
        
        if not csv_path.is_file():
            print(f"❌ Error: Not a file: {args.csv_file}")
            return False
        
        if csv_path.suffix.lower() != '.csv':
            print(f"⚠️  Warning: File doesn't have .csv extension: {args.csv_file}")
        
        print(f"✓ CSV file found: {csv_path}")
        
        # Check data dictionary if provided
        if args.data_dict:
            dict_path = Path(args.data_dict)
            if not dict_path.exists():
                print(f"❌ Error: Data dictionary not found: {args.data_dict}")
                return False
            
            if not dict_path.is_file():
                print(f"❌ Error: Not a file: {args.data_dict}")
                return False
            
            print(f"✓ Data dictionary found: {dict_path}")
        
        return True
    
    def load_data_dictionary(self, dict_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Load data dictionary from JSON file.
        
        Args:
            dict_path: Path to data dictionary JSON file
        
        Returns:
            Dictionary or None if not provided/failed to load
        """
        if not dict_path:
            return None
        
        try:
            with open(dict_path, 'r') as f:
                data_dict = json.load(f)
            
            print(f"✓ Loaded data dictionary with {len(data_dict.get('columns', {}))} columns")
            return data_dict
            
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in data dictionary: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading data dictionary: {e}")
            return None
    
    def display_data_dictionary(self, data_dict: Dict[str, Any]) -> None:
        """
        Display data dictionary information.
        
        Args:
            data_dict: Data dictionary to display
        """
        self.print_section("Data Dictionary Information")
        
        columns = data_dict.get("columns", {})
        print(f"Total columns defined: {len(columns)}\n")
        
        for col_name, col_def in columns.items():
            print(f"  • {col_name}")
            if "description" in col_def:
                print(f"    Description: {col_def['description']}")
            if "data_type" in col_def:
                print(f"    Type: {col_def['data_type']}")
            print()
    
    def display_progress(self, step: int, total: int, message: str) -> None:
        """
        Display progress update.
        
        Args:
            step: Current step number
            total: Total number of steps
            message: Progress message
        """
        percentage = (step / total) * 100
        bar_length = 40
        filled = int(bar_length * step / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r[{bar}] {percentage:.0f}% - {message}", end="", flush=True)
        
        if step == total:
            print()  # New line when complete
    
    def display_report(self, report: Dict[str, Any]) -> None:
        """
        Display final report in terminal.
        
        Args:
            report: Report data to display
        """
        formatted_report = report.get("formatted_reports", {}).get("text", "")
        
        if formatted_report:
            print("\n" + self.separator)
            print("ANALYSIS REPORT".center(self.width))
            print(self.separator)
            print(formatted_report)
        else:
            print("❌ Error: Report formatting failed")
    
    def save_report(
        self,
        report: Dict[str, Any],
        output_path: str,
        format: str
    ) -> bool:
        """
        Save report to file.
        
        Args:
            report: Report data
            output_path: Output file path
            format: Report format
        
        Returns:
            bool: True if saved successfully
        """
        try:
            formatted_reports = report.get("formatted_reports", {})
            content = formatted_reports.get(format, "")
            
            if not content:
                print(f"❌ Error: Report format '{format}' not available")
                return False
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Report saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return False
    
    def display_errors(self, errors: list) -> None:
        """
        Display errors.
        
        Args:
            errors: List of error messages
        """
        if errors:
            print("\n" + self.subseparator)
            print("❌ ERRORS:")
            print(self.subseparator)
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
    
    def display_warnings(self, warnings: list) -> None:
        """
        Display warnings.
        
        Args:
            warnings: List of warning messages
        """
        if warnings:
            print("\n" + self.subseparator)
            print("⚠️  WARNINGS:")
            print(self.subseparator)
            for i, warning in enumerate(warnings, 1):
                print(f"{i}. {warning}")
    
    def run(self) -> int:
        """
        Run the terminal interface.
        
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            # Print header
            self.print_header()
            
            # Parse arguments
            args = self.parse_arguments()
            
            # Setup logging
            self.setup_logging(args.verbose)
            
            # Validate files
            self.print_section("File Validation")
            if not self.validate_files(args):
                return 1
            
            # Load configuration
            self.print_section("Configuration")
            try:
                # Load .env file
                load_dotenv()
                
                # Get API key from environment or command line
                if args.api_key:
                    self.gemini_api_key = args.api_key
                    print("✓ Using API key from command line")
                else:
                    self.gemini_api_key = os.getenv('GEMINI_API_KEY')
                    if self.gemini_api_key:
                        print("✓ Using API key from .env file")
                
                if not self.gemini_api_key:
                    print("❌ Error: Gemini API key not found")
                    print("   Set GEMINI_API_KEY in .env file or use --api-key")
                    return 1
                
            except Exception as e:
                print(f"❌ Error loading configuration: {e}")
                return 1
            
            # Load data dictionary
            data_dict = self.load_data_dictionary(args.data_dict)
            if data_dict:
                self.display_data_dictionary(data_dict)
            
            # Initialize controller
            self.print_section("Initializing Agents")
            self.controller = MainController(
                gemini_api_key=self.gemini_api_key,
                log_level=logging.DEBUG if args.verbose else logging.INFO
            )
            
            if not self.controller.initialize_agents():
                print("❌ Error: Failed to initialize agents")
                return 1
            
            print("✓ All agents initialized successfully")
            
            # Execute workflow
            self.print_section("Executing Analysis Workflow")
            
            report_config = {
                "title": f"Analysis Report - {Path(args.csv_file).stem}",
                "export_formats": [args.format]
            }
            
            results = self.controller.execute_workflow(
                csv_path=args.csv_file,
                data_dict=data_dict,
                report_config=report_config
            )
            
            # Display results
            print()
            if results["success"]:
                print("✓ Analysis completed successfully!")
                print(f"  Execution time: {results['execution_time']:.2f} seconds")
                
                # Display warnings if any
                self.display_warnings(results.get("warnings", []))
                
                # Display or save report
                if args.output:
                    self.save_report(
                        results["report"],
                        args.output,
                        args.format
                    )
                else:
                    self.display_report(results["report"])
                
                return 0
            else:
                print("❌ Analysis failed")
                self.display_errors(results.get("errors", []))
                self.display_warnings(results.get("warnings", []))
                return 1
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Analysis interrupted by user")
            return 1
        except Exception as e:
            print(f"\n\n❌ Unexpected error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        finally:
            # Cleanup
            if self.controller:
                self.controller.cleanup()


def main():
    """Main entry point for terminal interface."""
    interface = TerminalInterface()
    exit_code = interface.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()