import serial
import struct
import time

def float_to_bytes_big_endian(value):
    return bytearray(struct.pack('>f', value))

def create_target_angle_packet(target_yaw, target_pitch):
    packet = bytearray()
    packet.extend([0xAA, 0x55])
    packet.extend(float_to_bytes_big_endian(target_yaw))
    packet.extend(float_to_bytes_big_endian(target_pitch))
    packet.extend([0x55, 0xAA])
    return packet

def send_target_angles(ser, target_yaw, target_pitch):
    packet = create_target_angle_packet(target_yaw, target_pitch)
    ser.write(packet)
    print(f"yaw: {target_yaw:.2f}°, pitch: {target_pitch:.2f}°")

def main():
    # 使用pySerial库
    ser = serial.Serial('/dev/ttyUSB0', 921600)
    
    try:
        angles = [
            (10.5, 20.3),
            (15.7, 25.2),
            (20.1, 30.8),
            (25.4, 35.6)
        ]
            
        for yaw, pitch in angles:
            send_target_angles(ser, yaw, pitch)
            time.sleep(0.1)
    
    finally:
        ser.close()
        print("串口已关闭")

if __name__ == "__main__":
    main()
