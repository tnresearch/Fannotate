# Build the image
docker build -t ui .

# Run the container with mounted volume
docker run -p 1337:1337 -v $(pwd)/uploads:/app/uploads ui
docker run -p 1337:1337 -v "$(pwd)/uploads:/app/uploads" ui
docker run -p 1337:1337 ui
