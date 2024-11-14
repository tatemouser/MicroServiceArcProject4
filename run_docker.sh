echo "Input filename to comment (first place in ./files directory): "
read fname

echo "Starting build process: "

FILENAME=$fname docker-compose build --no-cache
FILENAME=$fname docker-compose up

echo "Commenting is done."

docker-compose down