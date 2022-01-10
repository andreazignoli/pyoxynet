#!/bin/sh
aws lightsail create-container-service --service-name flask-service --power small --scale 1
aws lightsail push-container-image --service-name flask-service --label flask-pyoxynet --image flask-pyoxynet
aws lightsail create-container-service-deployment --service-name flask-service --containers file://containers.json --public-endpoint file://public-endpoint.json
# curl https://<<URL>>/
# curl -X POST https://<<URL>>/read_json -d @test_data/test_data.json
