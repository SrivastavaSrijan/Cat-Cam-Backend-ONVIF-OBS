"""
Response utilities for consistent API responses.
"""
from typing import Any, Dict, Optional


def create_api_response(
    data: Optional[Any] = None,
    success: Optional[str] = None,
    message: Optional[str] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a consistent API response format matching the TypeScript ApiResponse interface.
    
    Args:
        data: The response data
        success: Success message
        message: General message
        error: Error message
        
    Returns:
        Dictionary with consistent response format
    """
    response = {}
    
    if success is not None:
        response["success"] = success
    if message is not None:
        response["message"] = message
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
        
    return response


def success_response(data: Optional[Any] = None, message: Optional[str] = None) -> Dict[str, Any]:
    """Create a success response"""
    return create_api_response(data=data, success=message or "Operation completed successfully")


def error_response(error: str, message: Optional[str] = None) -> Dict[str, Any]:
    """Create an error response"""
    return create_api_response(error=error, message=message)


def data_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
    """Create a response with data"""
    return create_api_response(data=data, message=message)
