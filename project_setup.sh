#! /bin/bash

set -e

echo "Setting up Trading Data Pipeline..."

# Check if .env file exists, if not, create it from .env.example
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example"
    cp .env.example .env
    echo "IMPORTANT: Please fill in the credentials in the .env file"
    echo ""
fi

# Create virtual environment if it doesn't exist
if [ ! -d .venv ]; then
    echo "Creating virtual environment"
    python3 -m venv .venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment"
source .venv/bin/activate
echo "Virtual environment activated"

# Install dependencies
echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed"

# Generate Fernet key for airflow
echo "Generating Fernet key for airflow"
fernet_key=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
echo "Fernet key generated"

# Check if AIRFLOW_FERNET_KEY already exists
if grep -q "AIRFLOW_FERNET_KEY=" .env; then
    # Replace existing
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/AIRFLOW_FERNET_KEY=.*/AIRFLOW_FERNET_KEY=$fernet_key/" .env
    else
        sed -i "s/AIRFLOW_FERNET_KEY=.*/AIRFLOW_FERNET_KEY=$fernet_key/" .env
    fi
else
    # Append if doesn't exist
    echo "AIRFLOW_FERNET_KEY=$fernet_key" >> .env
fi

# Check if docker is running, if not, start it
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start docker and try again."
    exit 1
fi

# Start Docker services
echo "Starting Docker services"
docker compose up -d

echo "Waiting for services to be healthy..."
sleep 10

# Check if PostgreSQL is ready
echo "Checking PostgreSQL connection..."
max_attempts=10
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker exec trading-pipeline-postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo "PostgreSQL is ready"
        break
    fi
    attempt=$((attempt + 1))
    echo "PostgreSQL not ready yet... ($attempt/$max_attempts)"
    sleep 5
done

echo "Setup complete"
echo "Please fill in the credentials in the .env file"
echo "- Airflow UI at http://localhost:8083"