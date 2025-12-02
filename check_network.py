from network_utils import get_network_devices, ping_device, get_local_ip
import time

def scan_network():
    print("Scanning network for devices (this may take a few minutes)...")
    print("Your local IP address:", get_local_ip())
    print("\nLooking for devices on the network...")
    
    # Try scanning with different timeouts
    for timeout in [1.0, 2.0, 3.0]:
        print(f"\nScanning with timeout: {timeout} seconds...")
        devices = get_network_devices(timeout=timeout)
        
        if devices:
            print("\nFound the following devices:")
            print("-" * 70)
            print(f"{'IP Address':<15} | {'Hostname':<30} | Status")
            print("-" * 70)
            for ip, hostname, status in devices:
                print(f"{ip:<15} | {hostname:<30} | {status}")
            return devices
    
    print("\nNo devices found. Possible reasons:")
    print("1. Your friend's device might be blocking ICMP (ping) requests")
    print("2. The device might be on a different subnet")
    print("3. The device might have a firewall enabled")
    print("\nTry these troubleshooting steps:")
    print("- Make sure both devices are connected to the same WiFi network")
    print("- Ask your friend to temporarily disable their firewall")
    print("- Try connecting to your friend's device using their IP address directly")
    return []

def main():
    # First, try to find devices on the network
    devices = scan_network()
    
    if not devices:
        return
    
    # If we found devices, ask if user wants to ping a specific one
    while True:
        print("\nOptions:")
        print("1. Rescan network")
        print("2. Ping a specific IP address")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            devices = scan_network()
        elif choice == "2":
            target_ip = input("Enter IP address to ping: ").strip()
            print(f"\nPinging {target_ip}...")
            if ping_device(target_ip, count=4):
                print(f"Success! {target_ip} is reachable.")
            else:
                print(f"Failed to ping {target_ip}")
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()