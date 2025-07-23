import requests

pc_ip = "10.42.0.1"  # Replace with your PC's actual IP
response = requests.get(f"http://{pc_ip}:8001/api/endpoint")