FROM python:3.9-slim

# Create working directory in the container
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run the API inside the 'sentimnet' folder
CMD ["uvicorn", "sentimnet.app:app", "--host", "0.0.0.0", "--port", "8000"]
