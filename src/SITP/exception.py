import sys
from src.SITP.logger import logging


def error_message_detail(error: Exception, error_details: sys) -> str:
    """Generate a detailed error message.
    Args:
        error: The original exception instance.
        error_details: Typically the ``sys`` module used to retrieve traceback info.
    Returns:
        Formatted error string with file name, line number and message.
    """
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    return f"Error occurred in script [{file_name}] line [{line_no}]: {str(error)}"


class CustomException(Exception):
    """Custom exception that enriches the message with traceback details."""

    def __init__(self, error: Exception, error_details: sys):
        # Store the original error for potential downstream handling
        self.original_error = error
        # Build a detailed message using the helper above
        detailed_msg = error_message_detail(error, error_details)
        super().__init__(detailed_msg)
        self.error_message = detailed_msg

    def __str__(self) -> str:
        return self.error_message