#!/bin/bash

INSTALL_DIR="/opt/rpicam-surv"
CONFIG_DIR="/etc/rpicam-surv"
SERVICE_NAME="rpicam-surv"
EXECUTABLE_NAME="rpicam-surv"
EXECUTABLE_PATH=$(which rpicam-surv)

if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root (using sudo)."
  exit 1
fi

echo "Starting installation of the $SERVICE_NAME service..."

if systemctl list-units --all --type=service | grep -q "$SERVICE_NAME.service"; then
  echo "Stopping the old service..."
  systemctl stop $SERVICE_NAME
  systemctl disable $SERVICE_NAME
  rm -f /etc/systemd/system/$SERVICE_NAME.service
fi

echo "Cleaning up old files..."
rm -rf $INSTALL_DIR
rm -rf $CONFIG_DIR

echo "Installing new files..."
mkdir -p $INSTALL_DIR
cp $EXECUTABLE_PATH $INSTALL_DIR
chmod +x $INSTALL_DIR/$EXECUTABLE_NAME

echo "Creating the start.sh script..."
cat > $INSTALL_DIR/start.sh <<EOL
#!/bin/bash
log_file="$INSTALL_DIR/surv-\$(date +%Y-%m-%d_%H%M%S).log"
echo "--- Start: \$(date '+%Y-%m-%d %H:%M:%S') ---" >> \$log_file
$INSTALL_DIR/$EXECUTABLE_NAME \$(grep -E '^[^#]' $CONFIG_DIR/env.conf | tr '\\n' ' ') >> \$log_file 2>&1
EOL

chmod +x $INSTALL_DIR/start.sh

echo "Copy configuration files..."
mkdir -p $CONFIG_DIR
cp ./env.conf $CONFIG_DIR/env.conf

echo "Creating service files..."
cat > /etc/systemd/system/$SERVICE_NAME.service <<EOL
[Unit]
Description=Rpicam Surveillance Service
After=network.target

[Service]
ExecStart=$INSTALL_DIR/start.sh
WorkingDirectory=$INSTALL_DIR
Restart=always
User=$(whoami)
EnvironmentFile=$CONFIG_DIR/env.conf

[Install]
WantedBy=multi-user.target
EOL

echo "Enabling and starting the service..."
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME

echo -e "\n\e[32mThe service has been installed and started.\n\e[0m"
echo "You can customize the parameters by editing the $CONFIG_DIR/env.conf file."
echo "After making changes, execute the following command to apply the changes:"
echo "    sudo systemctl restart $SERVICE_NAME"
