# A general-purpose auto retry function.

def retry_with_exponential_backoff(func, max_retries=10, base_delay=1.0, operation_name="operation"):
    """
    Retry a function with exponential backoff for rate limit and other transient errors.
    
    Args:
        func: Function to retry (should be a callable with no arguments)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        operation_name: Name of the operation for logging
    
    Returns:
        Result of the function call
        
    Raises:
        The last exception encountered if all retries fail
    """
    import time

    for retry_attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__.lower()
            
            # Check if it's a retryable error
            retryable_errors = [
                'rate limit', 'timeout', 'temporary', 'service unavailable',
                'internal server error', 'bad gateway', 'service temporarily unavailable',
                'too many requests', 'quota', 'overloaded', 'resource has been exhausted',
                'resource_exhausted', 'ratelimiterror', 'quotaexceedederror',
                'connection error', 'network', 'json decode'
            ]
            
            # Also check specific litellm exceptions
            retryable_exception_types = [
                'ratelimiterror', 'timeouterror', 'apiconnectionerror', 
                'serviceunavailableerror', 'internalservererror', 'jsondecodeerror'
            ]
            
            is_retryable = (
                any(err in error_str for err in retryable_errors) or
                any(exc_type in error_type for exc_type in retryable_exception_types) or
                'code": 429' in error_str or  # HTTP 429 Too Many Requests
                'code": 503' in error_str or  # HTTP 503 Service Unavailable
                'code": 502' in error_str or  # HTTP 502 Bad Gateway
                'code": 500' in error_str     # HTTP 500 Internal Server Error
            )
            
            if retry_attempt == max_retries - 1:
                # Last attempt failed
                raise RuntimeError(f"{operation_name}: Failed after {max_retries} attempts. Error: {e}")
                
            elif is_retryable:
                # Special handling for rate limit errors - use longer delays
                is_rate_limit = (
                    'rate limit' in error_str or 'ratelimiterror' in error_type or
                    'quota' in error_str or 'resource has been exhausted' in error_str or
                    'code": 429' in error_str
                )
                
                if is_rate_limit:
                    # Longer delays for rate limits: 2, 8, 18, 32, 50 seconds
                    delay = 2 * (retry_attempt + 1) ** 2 + retry_attempt
                else:
                    # Standard exponential backoff for other errors
                    delay = base_delay * (2 ** retry_attempt) + (0.1 * retry_attempt)
                
                error_type_desc = "Rate limit" if is_rate_limit else "Retryable error"
                # print(f"{operation_name}: {error_type_desc} - Retry {retry_attempt + 1}/{max_retries} after {delay:.1f}s. Error: {e}")
                time.sleep(delay)
            else:
                # Non-retryable error
                print(f"{operation_name}: Non-retryable error: {e}")
                raise e
    
    # This should never be reached, but just in case
    raise RuntimeError(f"{operation_name}: Unexpected error - reached end of retry loop")