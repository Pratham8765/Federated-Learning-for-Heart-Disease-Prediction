import subprocess
import platform
import socket
import ipaddress
from typing import List, Optional, Tuple
import threading
import queue


def get_local_ip() -> str:
    """Get the local IP address of the current device."""
    try:
        # Create a socket connection to an external server (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))  # Google's public DNS server
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return "127.0.0.1"  # Fallback to localhost if unable to determine IP


def get_network_devices(ip_range: str = None, timeout: float = 1.0) -> List[Tuple[str, str, str]]:
    """
    Scan the local network for devices.
    Returns a list of tuples containing (ip, hostname, status).
    """
    if ip_range is None:
        # Default to the local subnet
        local_ip = get_local_ip()
        network = ".".join(local_ip.split(".")[:-1]) + ".0/24"
    else:
        network = ip_range

    devices = []
    results_queue = queue.Queue()

    def ping_host(ip):
        try:
            # Use system ping command
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '1', '-w', '1000', ip]
            response = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if response.returncode == 0:
                try:
                    hostname = socket.gethostbyaddr(ip)[0]
                except (socket.herror, socket.gaierror):
                    hostname = "Unknown"
                results_queue.put((ip, hostname, "Online"))
        except Exception as e:
            pass

    # Create and start threads for each IP in the network
    threads = []
    for ip in ipaddress.IPv4Network(network, strict=False):
        ip_str = str(ip)
        if ip_str.endswith('.0') or ip_str.endswith('.255'):
            continue  # Skip network and broadcast addresses
            
        t = threading.Thread(target=ping_host, args=(ip_str,))
        t.daemon = True
        threads.append(t)
        t.start()
        
        # Limit the number of concurrent threads
        if len(threads) >= 50:
            for t in threads:
                t.join(timeout=timeout)
            threads = []
    
    # Wait for remaining threads to complete
    for t in threads:
        t.join(timeout=timeout)
    
    # Get results from the queue
    while not results_queue.empty():
        devices.append(results_queue.get())
    
    return devices


def ping_device(ip: str, count: int = 4) -> bool:
    """
    Ping a specific IP address and return True if reachable.
    
    Args:
        ip: The IP address to ping
        count: Number of ping attempts
        
    Returns:
        bool: True if device is reachable, False otherwise
    """
    try:
        # Different ping commands for different operating systems
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, str(count), '-w', '1000', ip]
        response = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return response.returncode == 0
    except Exception as e:
        return False


def main():
    """Main function to demonstrate the network utilities."""
    print(f"Your local IP address: {get_local_ip()}")
    
    print("\nScanning local network for devices...")
    devices = get_network_devices()
    
    if not devices:
        print("No devices found on the network.")
        return
    
    print("\nFound the following devices:")
    print("-" * 50)
    print(f"{'IP Address':<15} | {'Hostname':<20} | Status")
    print("-" * 50)
    for ip, hostname, status in devices:
        print(f"{ip:<15} | {hostname:<20} | {status}")
    
    print("\nYou can ping any of these devices using the ping_device() function.")
    print("Example: ping_device('192.168.1.100')")


if __name__ == "__main__":
    main()
