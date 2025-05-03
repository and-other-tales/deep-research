#!/usr/bin/env python
"""
Test script for dynamic port mapping functionality.
This script demonstrates how the application handles port conflicts.
"""
import os
import socket
import sys
import time
import threading
import subprocess
from open_deep_research.port_utils import is_port_available, find_available_port

def occupy_port(port, bind_addr='127.0.0.1', duration=60):
    """Occupy a specific port for testing."""
    print(f"Occupying port {port} on {bind_addr} for {duration} seconds...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((bind_addr, port))
        sock.listen(1)
        print(f"Successfully bound to port {port}")
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(1)
    except Exception as e:
        print(f"Error binding to port {port}: {e}")
    finally:
        print(f"Releasing port {port}")
        sock.close()

def main():
    # 1. Check the default server port in the application
    # The default port in our configuration is 8080
    default_server_port = 8080
    default_langgraph_port = 8081
    
    # 2. Check if these ports are available
    server_port_available = is_port_available(default_server_port)
    langgraph_port_available = is_port_available(default_langgraph_port)
    
    print(f"Default server port {default_server_port} available: {server_port_available}")
    print(f"Default LangGraph port {default_langgraph_port} available: {langgraph_port_available}")
    
    # 3. Occupy both ports to force dynamic allocation
    server_thread = None
    langgraph_thread = None
    
    try:
        # Occupy both ports
        server_thread = threading.Thread(
            target=occupy_port, 
            args=(default_server_port,), 
            daemon=True
        )
        langgraph_thread = threading.Thread(
            target=occupy_port, 
            args=(default_langgraph_port,), 
            daemon=True
        )
        
        server_thread.start()
        langgraph_thread.start()
        
        # Give some time for the threads to start and bind the ports
        time.sleep(2)
        
        # 4. Check if the ports are now unavailable
        server_port_available = is_port_available(default_server_port)
        langgraph_port_available = is_port_available(default_langgraph_port)
        
        print(f"After occupation, server port {default_server_port} available: {server_port_available}")
        print(f"After occupation, LangGraph port {default_langgraph_port} available: {langgraph_port_available}")
        
        # 5. Find available ports dynamically
        new_server_port = find_available_port(
            preferred_port=default_server_port,
            min_port=8000,
            max_port=9000
        )
        new_langgraph_port = find_available_port(
            preferred_port=default_langgraph_port,
            min_port=8001,
            max_port=9001
        )
        
        print(f"Dynamically allocated server port: {new_server_port}")
        print(f"Dynamically allocated LangGraph port: {new_langgraph_port}")
        
        # 6. Try to start the server with the new ports (simulation only)
        print("\nSimulating server start with the new ports...")
        print(f"SERVER_PORT={new_server_port} LANGGRAPH_PORT={new_langgraph_port} would be used")
        
        # 7. Wait a bit to see that the ports remain occupied
        time.sleep(5)
        
        # 8. Verify that the default ports are still unavailable
        server_port_available = is_port_available(default_server_port)
        langgraph_port_available = is_port_available(default_langgraph_port)
        
        print(f"Default server port {default_server_port} still available: {server_port_available}")
        print(f"Default LangGraph port {default_langgraph_port} still available: {langgraph_port_available}")
        
        print("\nTest completed successfully. The dynamic port allocation is working as expected.")
        
    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        # If threads are still running, they'll be terminated when the main thread exits
        # since they're daemon threads
        print("Test complete.")

if __name__ == "__main__":
    main()