#!/bin/bash

echo "🚀 Starting Heroku container deployment..."

# Clean up Docker
echo "🧹 Cleaning up Docker system..."
docker system prune -a --volumes -f

# Login to Heroku
echo "📝 Logging into Heroku..."
heroku login

# Create app if it doesn't exist
APP_NAME="pyoxynet-lite-app"
echo "🏗️ Creating/checking Heroku app..."
heroku create $APP_NAME --stack=container 2>/dev/null || echo "App already exists or using existing app"

# Login to container registry
echo "🐳 Logging into Heroku container registry..."
heroku container:login 

# Build and push
echo "🔨 Building Docker image..."
docker build --no-cache -t registry.heroku.com/$APP_NAME/web . --platform linux/amd64

echo "📤 Pushing to Heroku..."
heroku container:push web -a $APP_NAME

echo "🚀 Releasing to Heroku..."
heroku container:release web -a $APP_NAME

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at: https://$APP_NAME.herokuapp.com"