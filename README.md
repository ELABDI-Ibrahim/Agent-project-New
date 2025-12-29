# Multi-Agent Data Analyzer

A sophisticated system that leverages specialized AI agents to process CSV data and generate comprehensive analytical reports.

## Project Structure

```
multi_agent_analyzer/
├── __init__.py                 # Package initialization and exports
├── config.py                   # Configuration management
├── core/                       # Core components
│   ├── __init__.py
│   ├── enums.py               # System enumerations
│   └── models.py              # Core data models
├── agents/                     # Agent implementations
│   └── __init__.py
└── utils/                      # Utility functions
    └── __init__.py

requirements.txt                # Python dependencies
setup.py                       # Package setup configuration
config.example.json            # Example configuration file
.env.example                   # Example environment variables
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

### Option 1: Environment Variables
Copy `.env.example` to `.env` and fill in your configuration:
```bash
cp .env.example .env
```

### Option 2: Configuration File
Copy `config.example.json` to `config.json` and customize:
```bash
cp config.example.json config.json
```

### Required Configuration
- **GEMINI_API_KEY**: Your Google Gemini API key (required)

## Core Components

### Data Models
- **AgentMessage**: Inter-agent communication protocol
- **DataProfile**: Dataset profiling information
- **StatisticalSummary**: Analysis results structure
- **Recommendation**: Decision agent output format
- **Report**: Final report structure
- **ColumnDefinition**: Data dictionary column definition
- **DataSchema**: Complete dataset schema

### Enumerations
- **MessageType**: REQUEST, RESPONSE, NOTIFICATION, ERROR
- **Priority**: HIGH, MEDIUM, LOW
- **DataType**: NUMERICAL, CATEGORICAL, TEMPORAL, TEXT, BOOLEAN
- **RiskLevel**: CRITICAL, HIGH, MEDIUM, LOW, MINIMAL
- **ExportFormat**: JSON, CSV, TXT, MARKDOWN
- **AnalysisState**: System state tracking

### Configuration Management
The `Config` class supports:
- Environment variable loading
- JSON file configuration
- Validation of settings
- Default value management

## Usage

```python
from multi_agent_analyzer import Config, AgentMessage, DataProfile

# Load configuration
config = Config.from_env()  # or Config.from_file("config.json")

# Validate configuration
config.validate()

# Use data models
message = AgentMessage(
    sender="collector",
    recipient="analyzer", 
    message_type=MessageType.REQUEST,
    content={"data": "sample"}
)
```

## Development

This project follows the multi-agent architecture pattern with:
- **Collector Agent**: Data preprocessing and validation
- **Analyzer Agent**: Statistical analysis and trend detection  
- **Decision Agent**: Recommendation generation
- **Reporter Agent**: Report synthesis and visualization
- **LLM Coordinator**: Gemini-powered orchestration

## Testing

The system uses property-based testing with Hypothesis:
- Minimum 100 iterations per property test
- Comprehensive input coverage through randomization
- Universal property validation across all inputs

## License

MIT License