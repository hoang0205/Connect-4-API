FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional but helps with some packages)
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Copy requirements file first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose port (optional but good practice)
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
