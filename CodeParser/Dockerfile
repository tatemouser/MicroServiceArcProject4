# Use Python as the base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy necessary files to the container
COPY *.py ./
COPY *.py ./files/




# Expose the port used for socket communication
EXPOSE 5000


# Run the parsing server script
CMD ["sh", "-c", "python3 CodeParser.py ${FILENAME}"]

