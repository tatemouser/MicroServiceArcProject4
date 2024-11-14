# Use Python as the base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy the adding comments client script and the original script
COPY ./*.py .
COPY ./*.py ./files/



# Run the adding comments client script
CMD ["sh", "-c", "python3 CommentPlacer.py ${FILENAME}"]
