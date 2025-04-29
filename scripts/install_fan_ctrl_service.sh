#!/bin/bash

SERVICE_NAME="fan_control.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
SCRIPT_NAME="fan_control.py"
SCRIPT_PATH="/usr/local/bin/$SCRIPT_NAME"

# Check if the service is installed
if systemctl list-unit-files | grep -q "$SERVICE_NAME"; then
    echo "Service $SERVICE_NAME is already installed. Reinstalling..."
    # Stop and disable the service before reinstalling
    sudo systemctl stop $SERVICE_NAME
    sudo systemctl disable $SERVICE_NAME
    sudo rm $SERVICE_PATH
fi

# Install python3-rpi.gpio if not already installed
if ! dpkg -s python3-rpi.gpio >/dev/null 2>&1; then
    echo "Installing required packages..."
    sudo apt-get update
    sudo apt-get install -y python3-rpi.gpio
else
    echo "Package python3-rpi.gpio is already installed. Skipping installation."
fi

# Copy script to /usr/local/bin
echo "Copying fan control script to $SCRIPT_PATH..."
sudo cp $SCRIPT_NAME $SCRIPT_PATH
sudo chmod +x $SCRIPT_PATH

# Create systemd file
echo "Creating systemd service file at $SERVICE_PATH..."
sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Fan Control Service
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 -u $SCRIPT_PATH
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOL

# Systemd reload
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable and start service
echo "Enabling and starting the fan control service..."
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

# Check service status
SERVICE_STATUS=$(sudo systemctl is-active $SERVICE_NAME)
if [ "$SERVICE_STATUS" = "active" ]; then
    echo "Service $SERVICE_NAME is installed and running."
else
    echo "Failed to start $SERVICE_NAME. Please check the logs."
fi
