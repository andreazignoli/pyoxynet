#!/bin/bash

# Deploy to Heroku script
# This script will login to Heroku and deploy the Flask app using containers

echo "🚀 Starting Heroku deployment process..."

# Login to Heroku
echo "📝 Logging into Heroku..."
heroku login

# Create Heroku app with container stack
echo "🏗️ Creating Heroku app..."
APP_NAME="pyoxynet-lite-$(date +%s)"
heroku create $APP_NAME --stack=container

if [ $? -ne 0 ]; then
    echo "❌ Failed to create Heroku app. Trying with existing app..."
    read -p "Enter your Heroku app name: " APP_NAME
fi

# Set stack to container (in case it's not already set)
echo "🐳 Setting stack to container..."
heroku stack:set container -a $APP_NAME

# Add Heroku git remote
echo "🔗 Adding Heroku git remote..."
heroku git:remote -a $APP_NAME

# Commit any pending changes
echo "📦 Committing changes..."
git add .
git commit -m "Update for Heroku deployment

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>" || echo "No changes to commit"

# Deploy to Heroku
echo "🚢 Deploying to Heroku..."
git push heroku HEAD:main

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Your app is available at: https://$APP_NAME.herokuapp.com"
    
    # Open the app in browser (optional)
    read -p "Open app in browser? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        heroku open -a $APP_NAME
    fi
else
    echo "❌ Deployment failed. Check the logs with: heroku logs --tail -a $APP_NAME"
fi

echo "📋 Useful commands:"
echo "  - View logs: heroku logs --tail -a $APP_NAME"
echo "  - Open app: heroku open -a $APP_NAME"
echo "  - Scale dynos: heroku ps:scale web=1 -a $APP_NAME"