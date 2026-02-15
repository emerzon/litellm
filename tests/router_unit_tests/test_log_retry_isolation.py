"""
Test cases for Router.log_retry() cross-request contamination fix
"""
import pytest
from litellm import Router


def test_log_retry_no_cross_request_contamination():
    """Test that different requests have independent previous_models lists"""
    model_list = [
        {
            'model_name': 'gpt-model',
            'litellm_params': {
                'model': 'openai/gpt-fake',
                'api_key': 'fake-key-1',
            },
            'model_info': {'id': 'deployment-1'},
        },
    ]
    
    router = Router(model_list=model_list, num_retries=2)
    
    # Simulate two different requests retrying
    exc_a = Exception('Request A: rate limit')
    kwargs_a = {'model': 'gpt-model', 'metadata': {'request_id': 'req-A'}}
    router.log_retry(kwargs=kwargs_a, e=exc_a)
    
    exc_b = Exception('Request B: timeout')
    kwargs_b = {'model': 'gpt-model', 'metadata': {'request_id': 'req-B'}}
    router.log_retry(kwargs=kwargs_b, e=exc_b)
    
    # Each request should have its own independent list
    pm_a = kwargs_a['metadata']['previous_models']
    pm_b = kwargs_b['metadata']['previous_models']
    
    # They should NOT be the same object
    assert pm_a is not pm_b, "Request A and B should have independent previous_models lists"
    
    # Each should contain only their own retry
    assert len(pm_a) == 1, f"Request A should have 1 retry, got {len(pm_a)}"
    assert len(pm_b) == 1, f"Request B should have 1 retry, got {len(pm_b)}"
    
    # Verify the content is correct
    assert pm_a[0]['metadata']['request_id'] == 'req-A'
    assert pm_b[0]['metadata']['request_id'] == 'req-B'


def test_log_retry_per_request_cap():
    """Test that the 3-entry cap applies per request, not globally"""
    model_list = [
        {
            'model_name': 'gpt-model',
            'litellm_params': {
                'model': 'openai/gpt-fake',
                'api_key': 'fake-key',
            },
            'model_info': {'id': 'd1'},
        },
    ]
    
    router = Router(model_list=model_list, num_retries=5)
    
    # Single request with 5 retries
    kwargs = {'model': 'gpt-model', 'metadata': {'request_id': 'req-single'}}
    
    for i in range(5):
        exc = Exception(f'Retry {i}')
        router.log_retry(kwargs=kwargs, e=exc)
    
    pm = kwargs['metadata']['previous_models']
    
    # Should be capped at 3 entries for this request
    assert len(pm) == 3, f"previous_models should be capped at 3, got {len(pm)}"
    
    # Should have the last 3 retries (indices 2, 3, 4)
    assert 'Retry 2' in pm[0]['exception_string']
    assert 'Retry 3' in pm[1]['exception_string']
    assert 'Retry 4' in pm[2]['exception_string']


def test_log_retry_concurrent_requests_independent():
    """Test that concurrent requests with multiple retries don't interfere"""
    model_list = [
        {
            'model_name': 'gpt-model',
            'litellm_params': {
                'model': 'openai/gpt-fake',
                'api_key': 'fake-key',
            },
            'model_info': {'id': 'd1'},
        },
    ]
    
    router = Router(model_list=model_list, num_retries=5)
    
    # Create 3 different requests, each with multiple retries
    kwargs_list = [
        {'model': 'gpt-model', 'metadata': {'request_id': f'req-{i}'}}
        for i in range(3)
    ]
    
    # Simulate interleaved retries
    for i in range(4):
        for j, kwargs in enumerate(kwargs_list):
            exc = Exception(f'Request {j} retry {i}')
            router.log_retry(kwargs=kwargs, e=exc)
    
    # Each request should have exactly 3 entries (capped)
    for j, kwargs in enumerate(kwargs_list):
        pm = kwargs['metadata']['previous_models']
        assert len(pm) == 3, f"Request {j} should have 3 retries, got {len(pm)}"
        
        # Verify all entries belong to the same request
        for entry in pm:
            assert entry['metadata']['request_id'] == f'req-{j}'


def test_log_retry_no_api_key_leakage():
    """Test that API keys from one request don't leak into another"""
    model_list = [
        {
            'model_name': 'gpt-model',
            'litellm_params': {
                'model': 'openai/gpt-fake',
                'api_key': 'fake-key',
            },
            'model_info': {'id': 'd1'},
        },
    ]
    
    router = Router(model_list=model_list, num_retries=2)
    
    # Tenant A's request
    kwargs_a = {
        'model': 'gpt-model',
        'api_key': 'sk-tenant-A-secret',
        'metadata': {'user_api_key': 'tenant-A-key', 'request_id': 'req-A'}
    }
    router.log_retry(kwargs=kwargs_a, e=Exception('error A'))
    
    # Tenant B's request
    kwargs_b = {
        'model': 'gpt-model',
        'api_key': 'sk-tenant-B-secret',
        'metadata': {'user_api_key': 'tenant-B-key', 'request_id': 'req-B'}
    }
    router.log_retry(kwargs=kwargs_b, e=Exception('error B'))
    
    # Verify tenant B can't see tenant A's keys
    pm_b = kwargs_b['metadata']['previous_models']
    assert len(pm_b) == 1, f"Tenant B should only see their own retry"
    assert pm_b[0]['api_key'] == 'sk-tenant-B-secret'
    assert pm_b[0]['metadata']['user_api_key'] == 'tenant-B-key'
    assert pm_b[0]['metadata']['request_id'] == 'req-B'
    
    # Verify no tenant A data in tenant B's list
    for entry in pm_b:
        assert 'tenant-A' not in str(entry)


def test_log_retry_with_litellm_metadata():
    """Test that log_retry works correctly with litellm_metadata instead of metadata"""
    model_list = [
        {
            'model_name': 'gpt-model',
            'litellm_params': {
                'model': 'openai/gpt-fake',
                'api_key': 'fake-key',
            },
            'model_info': {'id': 'd1'},
        },
    ]
    
    router = Router(model_list=model_list, num_retries=2)
    
    # Use litellm_metadata instead of metadata
    kwargs_a = {'model': 'gpt-model', 'litellm_metadata': {'request_id': 'req-A'}}
    router.log_retry(kwargs=kwargs_a, e=Exception('error A'))
    
    kwargs_b = {'model': 'gpt-model', 'litellm_metadata': {'request_id': 'req-B'}}
    router.log_retry(kwargs=kwargs_b, e=Exception('error B'))
    
    # Both should have independent lists
    pm_a = kwargs_a['litellm_metadata']['previous_models']
    pm_b = kwargs_b['litellm_metadata']['previous_models']
    
    assert pm_a is not pm_b
    assert len(pm_a) == 1
    assert len(pm_b) == 1
    assert pm_a[0]['litellm_metadata']['request_id'] == 'req-A'
    assert pm_b[0]['litellm_metadata']['request_id'] == 'req-B'
