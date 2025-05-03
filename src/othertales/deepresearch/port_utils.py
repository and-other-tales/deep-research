"""
Utility functions for dynamic port mapping and management.

This module provides tools to:
1. Find available ports on the system
2. Handle port conflicts gracefully
3. Support dynamic port allocation when static ports are not available
"""

import os
import socket
import logging
import random
from typing import Optional, List, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

def is_port_available(port: int, host: str = '127.0.0.1') -> bool:
    """
    Check if a port is available on the specified host.
    
    Args:
        port: The port number to check
        host: The host to check (default: 127.0.0.1)
        
    Returns:
        bool: True if the port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)  # Set a short timeout for connection attempts
            result = s.connect_ex((host, port))
            return result != 0  # If result is 0, the port is in use
    except (socket.error, OSError) as e:
        logger.warning(f"Error checking if port {port} is available: {e}")
        return False  # Assume port is not available if there's an error

def find_available_port(
    preferred_port: Optional[int] = None,
    min_port: int = 8000,
    max_port: int = 9000,
    host: str = '127.0.0.1'
) -> int:
    """
    Find an available port on the system.
    
    The function will try to use the preferred port first, if provided.
    If that port is not available, it will search for an available port
    in the specified range.
    
    Args:
        preferred_port: The port to try first (optional)
        min_port: The minimum port number to check (default: 8000)
        max_port: The maximum port number to check (default: 9000)
        host: The host to check (default: 127.0.0.1)
        
    Returns:
        int: An available port
        
    Raises:
        RuntimeError: If no available port could be found
    """
    # Try the preferred port first if provided
    if preferred_port is not None:
        if is_port_available(preferred_port, host):
            logger.info(f"Preferred port {preferred_port} is available")
            return preferred_port
        else:
            logger.info(f"Preferred port {preferred_port} is not available, searching for alternative")

    # Try random ports in the range to avoid collisions when multiple
    # instances are started at the same time
    port_range = list(range(min_port, max_port + 1))
    random.shuffle(port_range)
    
    for port in port_range:
        if is_port_available(port, host):
            logger.info(f"Found available port: {port}")
            return port
    
    # If we reach here, no port was available
    raise RuntimeError(f"Could not find an available port in the range {min_port} to {max_port}")

def get_port_from_env(
    env_var: str,
    default_port: int,
    fallback_min: int = 8000,
    fallback_max: int = 9000,
    host: str = '127.0.0.1'
) -> int:
    """
    Get a port number from an environment variable with fallback to dynamic allocation.
    
    Args:
        env_var: The name of the environment variable to check
        default_port: The default port to use if the environment variable is not set
        fallback_min: The minimum port number to use if dynamic allocation is needed
        fallback_max: The maximum port number to use if dynamic allocation is needed
        host: The host to check (default: 127.0.0.1)
        
    Returns:
        int: The port number to use
    """
    # Try to get port from environment variable
    port_str = os.environ.get(env_var)
    
    if port_str:
        try:
            # Try to convert to integer
            preferred_port = int(port_str)
            
            # Check if the port is available
            if is_port_available(preferred_port, host):
                return preferred_port
            else:
                logger.warning(
                    f"Port {preferred_port} from environment variable {env_var} "
                    f"is not available. Finding alternative port."
                )
        except ValueError:
            logger.warning(
                f"Invalid port number '{port_str}' in environment variable {env_var}. "
                f"Using dynamic port allocation."
            )
    
    # If we didn't get a valid port from the environment variable or it's not available,
    # try to use the default port
    if is_port_available(default_port, host):
        return default_port
    
    # If the default port is not available, find a random available port
    return find_available_port(
        preferred_port=default_port,
        min_port=fallback_min,
        max_port=fallback_max,
        host=host
    )

def allocate_port_range(
    count: int,
    preferred_start: Optional[int] = None,
    min_port: int = 8000,
    max_port: int = 9000,
    host: str = '127.0.0.1'
) -> List[int]:
    """
    Allocate a contiguous range of available ports.
    
    Args:
        count: The number of consecutive ports needed
        preferred_start: The preferred starting port (optional)
        min_port: The minimum port number to check (default: 8000)
        max_port: The maximum port number to check (default: 9000)
        host: The host to check (default: 127.0.0.1)
        
    Returns:
        List[int]: A list of consecutive available ports
        
    Raises:
        RuntimeError: If the requested number of consecutive ports could not be found
    """
    if count <= 0:
        raise ValueError("Count must be a positive integer")
    
    # Try the preferred range first
    if preferred_start is not None:
        if all(is_port_available(preferred_start + i, host) for i in range(count)):
            return [preferred_start + i for i in range(count)]
    
    # Search for a contiguous range of available ports
    for start_port in range(min_port, max_port - count + 2):
        # Check if all ports in the range are available
        if all(is_port_available(start_port + i, host) for i in range(count)):
            return [start_port + i for i in range(count)]
    
    # If we reach here, no contiguous range was found
    raise RuntimeError(f"Could not find {count} consecutive available ports in the range {min_port} to {max_port}")