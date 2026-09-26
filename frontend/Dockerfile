# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy ONLY requirements first so dependencies install cleanly 
COPY requirements.txt .

# Install Python dependencies without caching leftovers
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application code afterwards
COPY . .

# Define the command to run the Streamlit app properly bound to HF network routing
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
