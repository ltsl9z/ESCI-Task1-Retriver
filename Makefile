# Define variables
VENV_NAME = .venv
PYTHON = python3
REQUIREMENTS = requirements.txt

# Create a virtual environment
create_env:
	$(PYTHON) -m venv $(VENV_NAME)

# Install dependencies
install_deps: create_env
	$(VENV_NAME)/bin/pip install -r $(REQUIREMENTS)

# Clean up the virtual environment
clean:
	rm -rf $(VENV_NAME)

# Default target
all: install_deps
