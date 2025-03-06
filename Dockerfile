# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install git and other build dependencies
RUN apt-get update && \
    apt-get install -y git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Create a directory for Streamlit configuration
RUN mkdir -p /root/.streamlit

# Create Streamlit config with server settings
RUN echo '\
    [server]\n\
    port = 8501\n\
    address = "*"\n\
    headless = true\n\
    enableCORS = false\n\
    enableXsrfProtection = false\n\
    ' > /root/.streamlit/config.toml

# Command to run the application
CMD ["streamlit", "run", "app.py"] 