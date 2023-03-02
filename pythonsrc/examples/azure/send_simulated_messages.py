import asyncio
import random
from azure.iot.device import Message
from azure.iot.device.aio import IoTHubDeviceClient

# CONNECTION_STRING = "HostName=testhubname0.azure-devices.net;DeviceId=RPiDevice0;SharedAccessKey=hv82yeuIj9GTTXnAhuw8HWk7NGUOp9NU4NO/0NOMb9E="

# CONNECTION_STRING = "HostName=testhubname0.azure-devices.net;DeviceId=RPiDevice0;SharedAccessKey=hv82yeuIj9GTTXnAhuw8HWk7NGUOp9NU4NO/0NOMb9E="

CONNECTION_STRING = "HostName=sonydemoiothub.azure-devices.net;DeviceId=sonytestdevice;SharedAccessKey=p/c31bhBeVelkeYLcx7Bws5BY5tBYXym3v1Tg7xTK/U="

TEMPERATURE = 20.0
HUMIDITY = 33
PAYLOAD = '{{"temperature": {temperature}, "humidity": {humidity}}}'

async def main():

    try:
        # Create instance of the device client
        client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)

        print("Simulated device started. Press Ctrl-C to exit")
        while True:

            temperature = round(TEMPERATURE, 2)
            humidity = round(HUMIDITY + (random.random() * 20), 2)
            data = PAYLOAD.format(temperature=temperature, humidity=humidity)
            message = Message(data)

            # Send a message to the IoT hub
            print(f"Sending message: {message}")
            await client.send_message(message)
            print("Message successfully sent")

            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("Simulated device stopped")

if __name__ == '__main__':
    asyncio.run(main())
