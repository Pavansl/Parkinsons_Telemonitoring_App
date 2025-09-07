# Use official Python image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy all files and folders into the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so internal app module is importable
ENV PYTHONPATH=/app

# Default command to keep container running
CMD ["tail", "-f", "/dev/null"]
