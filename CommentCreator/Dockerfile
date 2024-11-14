# Use Python as the base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy the commenting server script
COPY CommentCreator.py ./



# No need to install any dependencies

# Expose the port used for server
EXPOSE 5001

# Run the commenting server script
CMD ["sh", "-c", "python3 CommentCreator.py ${FILENAME}"]
