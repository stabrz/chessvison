# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /

# Copy the requirements file into the container
COPY requirements.txt .

# # Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Command to run the application
CMD ["python", "API.py"]