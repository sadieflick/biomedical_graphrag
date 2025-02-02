## To Run with Docker
# Build and start all services
cd docker
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down