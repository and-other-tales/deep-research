"""
Tests for the port_utils module.
"""
import os
import socket
import threading
import time
import pytest
from unittest.mock import patch
from othertales.deepresearch.port_utils import (
    is_port_available,
    find_available_port,
    get_port_from_env,
    allocate_port_range
)

def occupy_port(port, duration=1):
    """Helper function to occupy a port for testing."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', port))
    sock.listen(1)
    time.sleep(duration)
    sock.close()

def test_is_port_available():
    """Test the is_port_available function."""
    # Pick a port unlikely to be in use
    test_port = 8765
    
    # Test with a port that should be available
    assert is_port_available(test_port) is True, "Port should be available"
    
    # Temporarily occupy the port in a separate thread
    thread = threading.Thread(target=occupy_port, args=(test_port,))
    thread.daemon = True
    thread.start()
    time.sleep(0.1)  # Give time for the thread to start and bind the port
    
    # Test with a port that should be unavailable
    assert is_port_available(test_port) is False, "Port should be unavailable"
    
    # Wait for the thread to complete
    thread.join()

def test_find_available_port():
    """Test the find_available_port function."""
    # Find an available port
    port1 = find_available_port(min_port=8700, max_port=8799)
    assert 8700 <= port1 <= 8799, "Port should be in the specified range"
    
    # Temporarily occupy this port
    thread = threading.Thread(target=occupy_port, args=(port1,))
    thread.daemon = True
    thread.start()
    time.sleep(0.1)  # Give time for the thread to start and bind the port
    
    # Find another port, which should be different
    port2 = find_available_port(min_port=8700, max_port=8799)
    assert port1 != port2, "Different available port should be found"
    
    # Wait for the thread to complete
    thread.join()

def test_find_available_port_with_preferred():
    """Test the find_available_port function with a preferred port."""
    # Find an available port
    free_port = find_available_port(min_port=8700, max_port=8799)
    
    # Request the same port as preferred
    port = find_available_port(preferred_port=free_port, min_port=8700, max_port=8799)
    assert port == free_port, "Should return the preferred port if available"
    
    # Temporarily occupy this port
    thread = threading.Thread(target=occupy_port, args=(free_port,))
    thread.daemon = True
    thread.start()
    time.sleep(0.1)  # Give time for the thread to start and bind the port
    
    # Request the same port as preferred, but it's occupied
    port = find_available_port(preferred_port=free_port, min_port=8700, max_port=8799)
    assert port != free_port, "Should return a different port if preferred port is not available"
    
    # Wait for the thread to complete
    thread.join()

def test_get_port_from_env():
    """Test the get_port_from_env function."""
    # Test without environment variable set
    default_port = 8888
    port = get_port_from_env("TEST_PORT", default_port)
    
    # If default_port is available, it should be returned; otherwise, another port should be returned
    if is_port_available(default_port):
        assert port == default_port, "Should return default port if available"
    else:
        assert port != default_port, "Should return different port if default port is not available"
    
    # Test with environment variable set
    with patch.dict(os.environ, {"TEST_PORT": "8999"}):
        port = get_port_from_env("TEST_PORT", default_port)
        
        # If 8999 is available, it should be returned; otherwise, another port should be returned
        if is_port_available(8999):
            assert port == 8999, "Should return port from environment variable if available"
        else:
            assert port != 8999, "Should return different port if port from environment variable is not available"
    
    # Test with invalid environment variable
    with patch.dict(os.environ, {"TEST_PORT": "invalid"}):
        port = get_port_from_env("TEST_PORT", default_port)
        
        # If default_port is available, it should be returned; otherwise, another port should be returned
        if is_port_available(default_port):
            assert port == default_port, "Should return default port if environment variable is invalid"
        else:
            assert port != default_port, "Should return different port if default port is not available"

def test_allocate_port_range():
    """Test the allocate_port_range function."""
    # Allocate a range of 3 ports
    ports = allocate_port_range(3, min_port=8800, max_port=8899)
    assert len(ports) == 3, "Should return exactly 3 ports"
    assert all(8800 <= port <= 8899 for port in ports), "All ports should be in the specified range"
    assert ports[1] == ports[0] + 1 and ports[2] == ports[1] + 1, "Ports should be consecutive"
    
    # Temporarily occupy the middle port
    thread = threading.Thread(target=occupy_port, args=(ports[1],))
    thread.daemon = True
    thread.start()
    time.sleep(0.1)  # Give time for the thread to start and bind the port
    
    # Allocate another range, which should be different
    new_ports = allocate_port_range(3, min_port=8800, max_port=8899)
    assert set(ports).intersection(set(new_ports)) == set(), "Should find a completely different range"
    assert new_ports[1] == new_ports[0] + 1 and new_ports[2] == new_ports[1] + 1, "New ports should be consecutive"
    
    # Wait for the thread to complete
    thread.join()