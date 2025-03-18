FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Create a directory for output files
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
#
#
#
#
#
#switch this so that it runs all the files in the src in the correct sequential order
ENTRYPOINT ["python", "src/dewatering-tool.py"]