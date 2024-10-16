import RPi.GPIO as GPIO
import time
import signal
import sys

FAN_GPIO = 14

TEMP_THRESHOLD_ON = 50.0
TEMP_THRESHOLD_OFF = 45.0


def get_cpu_temperature():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            cpu_temp = int(f.read()) / 1000.0
        return cpu_temp
    except FileNotFoundError:
        print("Error: Could not read CPU temperature.")
        return 0.0

def cleanup_gpio():
    GPIO.output(FAN_GPIO, GPIO.LOW)
    GPIO.cleanup()
    print("GPIO cleanup and fan turned off.")

def handle_signal(signum, frame):
    print(f"Received signal {signum}, cleaning up GPIO...")
    cleanup_gpio()
    exit(0)

def control_fan():
    fan_on = False

    while True:
        cpu_temp = get_cpu_temperature()
        print(f"CPU Temperature: {cpu_temp}C")

        if cpu_temp > TEMP_THRESHOLD_ON and not fan_on:
            print(f"Turn on fan...")
            GPIO.output(FAN_GPIO, GPIO.HIGH)
            fan_on = True
        elif cpu_temp < TEMP_THRESHOLD_OFF and fan_on:
            print(f"Turn off fan...")
            GPIO.output(FAN_GPIO, GPIO.LOW)
            fan_on = False

        # Check temp every 5 seconds
        time.sleep(5)


if __name__ == "__main__":
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(FAN_GPIO, GPIO.OUT)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    control_fan()
