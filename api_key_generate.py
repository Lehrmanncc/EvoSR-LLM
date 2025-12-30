import requests
import os
from typing import List, Dict


def generate_api_keys(
    start_key_name: str,
    num_keys: int,
    api_url: str = "https://aihubmix.com",
    access_token: str = None,
    remain_quota: int = 0,
    expired_time: int = -1,
    unlimited_quota: bool = True,
    subnet: str = "",
    output_file: str = ".env"
) -> List[Dict[str, str]]:
    """
    Batch generate API keys
    
    Args:
        start_key_name: Starting key name (numeric string like "0", "5")
        num_keys: Number of keys to generate
        api_url: API base URL
        access_token: Access token
        remain_quota: Remaining quota
        expired_time: Expiration time (-1 for never expires)
        unlimited_quota: Whether to use unlimited quota
        subnet: Subnet restriction
        output_file: Output file name (default: .env)
    
    Returns:
        List of generated API keys with {'name': ..., 'key': ..., 'id': ...}
    """
    
    if access_token is None:
        raise ValueError("access_token is required")
    
    try:
        start_num = int(start_key_name)
    except ValueError:
        raise ValueError(f"start_key_name must be a numeric string, got: {start_key_name}")
    
    headers = {
        "Authorization": access_token,
        "Content-Type": "application/json"
    }
    
    generated_keys = []
    failed_keys = []
    
    print(f"Generating {num_keys} API keys starting from {start_key_name}...")
    
    for i in range(num_keys):
        current_key_name = str(start_num + i)
        
        # For unlimited quota, don't set remain_quota
        if unlimited_quota:
            payload = {
                "name": current_key_name,
                "expired_time": expired_time,
                "unlimited_quota": True,
                "subnet": subnet
            }
        else:
            payload = {
                "name": current_key_name,
                "expired_time": expired_time,
                "remain_quota": remain_quota,
                "unlimited_quota": False,
                "subnet": subnet
            }
        
        try:
            response = requests.post(f"{api_url}/api/token/", headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success", False):
                    token_data = data.get("data", {})
                    key_info = {
                        'name': current_key_name,
                        'key': token_data.get('key', '未知'),
                        'id': token_data.get('id', '未知')
                    }
                    generated_keys.append(key_info)
                    quota_info = "unlimited" if unlimited_quota else f"quota: {remain_quota}"
                    print(f"✓ [{i+1}/{num_keys}] {current_key_name} ({quota_info})")
                else:
                    error_msg = data.get('message', 'unknown error')
                    print(f"✗ [{i+1}/{num_keys}] {current_key_name} - {error_msg}")
                    failed_keys.append({'name': current_key_name, 'error': error_msg})
            else:
                error_msg = f"status {response.status_code}"
                print(f"✗ [{i+1}/{num_keys}] {current_key_name} - {error_msg}")
                failed_keys.append({'name': current_key_name, 'error': error_msg})
                
        except Exception as e:
            print(f"✗ [{i+1}/{num_keys}] {current_key_name} - {str(e)}")
            failed_keys.append({'name': current_key_name, 'error': str(e)})
    
    print(f"\nCompleted: {len(generated_keys)}/{num_keys} successful")
    if failed_keys:
        print(f"Failed: {len(failed_keys)}")
    
    # 保存到.env文件
    if generated_keys:
        save_to_env_file(generated_keys, output_file)
    
    return generated_keys


def save_to_env_file(keys: List[Dict[str, str]], output_file: str = ".env"):
    """
    Save generated API keys to .env file
    
    Args:
        keys: List of API keys
        output_file: Output file path
    """
    print(f"\nSaving to {output_file}...")
    
    # Backup existing file
    if os.path.exists(output_file):
        backup_file = output_file + ".backup"
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            with open(backup_file, 'w') as f:
                f.write(content)
        except Exception as e:
            print(f"Backup failed: {e}")
    
    # Write to .env file starting from API_KEY_0
    try:
        with open(output_file, 'w') as f:
            for idx, key_info in enumerate(keys):
                env_var_name = f"API_KEY_{idx}"
                # 确保key是字符串，并且以sk-开头
                key_value = str(key_info['key'])
                if not key_value.startswith('sk-'):
                    key_value = f"sk-{key_value}"
                # Wrap key value in quotes
                f.write(f'{env_var_name}="{key_value}"\n')
        
        print(f"✓ Saved {len(keys)} keys to {output_file}")
        
    except Exception as e:
        print(f"✗ Save failed: {e}")


def query_existing_keys(
    start_key_name: str,
    num_keys: int,
    api_url: str = "https://aihubmix.com",
    access_token: str = None,
    output_file: str = ".env"
) -> List[Dict[str, str]]:
    """
    Query existing API keys and save to .env file
    
    Args:
        start_key_name: Starting key name (numeric string)
        num_keys: Number of keys to query
        api_url: API base URL
        access_token: Access token
        output_file: Output file name
    
    Returns:
        List of queried API keys
    """
    
    if access_token is None:
        raise ValueError("access_token is required")
    
    try:
        start_num = int(start_key_name)
    except ValueError:
        raise ValueError(f"start_key_name must be numeric string, got: {start_key_name}")
    
    headers = {
        "Authorization": access_token,
        "Content-Type": "application/json"
    }
    
    print(f"Querying {num_keys} keys starting from {start_key_name}...")
    
    try:
        # Get all keys with pagination
        all_tokens = []
        page = 0
        page_size = 10
        
        while True:
            params = {'p': page, 'size': page_size}
            response = requests.get(f"{api_url}/api/token/", headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"✗ Request failed: status {response.status_code}")
                break
            
            data = response.json()
            if not data.get("success", False):
                break
            
            page_tokens = data.get("data", [])
            if not page_tokens:
                break
            
            all_tokens.extend(page_tokens)
            
            if len(page_tokens) < page_size:
                break
            
            page += 1
        
        print(f"Found {len(all_tokens)} total keys")
        
        # Create name to token mapping
        tokens_dict = {str(token.get('name', '')): token for token in all_tokens}
        
        found_keys = []
        missing_keys = []
        
        for i in range(num_keys):
            current_key_name = str(start_num + i)
            
            if current_key_name in tokens_dict:
                token = tokens_dict[current_key_name]
                key_info = {
                    'name': current_key_name,
                    'key': str(token.get('key', '')),
                    'id': token.get('id', 'unknown')
                }
                found_keys.append(key_info)
            else:
                missing_keys.append(current_key_name)
        
        print(f"\nFound: {len(found_keys)}/{num_keys}")
        if missing_keys:
            print(f"Missing: {', '.join(missing_keys)}")
        
        if found_keys:
            save_to_env_file(found_keys, output_file)
        else:
            print("No keys found")
        
        return found_keys
        
    except Exception as e:
        print(f"✗ Query failed: {str(e)}")
        return []


def list_all_keys(
    api_url: str = "https://aihubmix.com",
    access_token: str = None
) -> List[Dict[str, str]]:
    """
    List all available API keys
    
    Args:
        api_url: API base URL
        access_token: Access token
    
    Returns:
        List of all API keys
    """
    if access_token is None:
        raise ValueError("access_token is required")
    
    headers = {
        "Authorization": access_token,
        "Content-Type": "application/json"
    }
    
    print("Fetching all API keys...")
    
    try:
        all_tokens = []
        page = 0
        page_size = 10
        
        while True:
            params = {'p': page, 'size': page_size}
            response = requests.get(f"{api_url}/api/token/", headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"✗ Request failed: status {response.status_code}")
                break
            
            data = response.json()
            if not data.get("success", False):
                break
            
            page_tokens = data.get("data", [])
            if not page_tokens:
                break
            
            all_tokens.extend(page_tokens)
            
            if len(page_tokens) < page_size:
                break
            
            page += 1
        
        print(f"Found {len(all_tokens)} keys\n")
        
        if all_tokens:
            sorted_tokens = sorted(all_tokens, key=lambda x: int(x.get('name', '0')) if str(x.get('name', '')).isdigit() else 999999)
            
            for idx, token in enumerate(sorted_tokens, 1):
                name = token.get('name', 'unknown')
                key = str(token.get('key', ''))
                token_id = token.get('id', 'unknown')
                
                if not key.startswith('sk-'):
                    key = f"sk-{key}"
                
                masked_key = key[:10] + "..." + key[-5:] if len(key) > 15 else "***"
                print(f"  [{idx}] {name:>3} | ID: {token_id} | {masked_key}")
            
            numeric_names = sorted([int(t.get('name', '0')) for t in all_tokens if str(t.get('name', '')).isdigit()])
            if numeric_names:
                print(f"\nRange: {numeric_names[0]}-{numeric_names[-1]} ({len(numeric_names)} keys)")
        return all_tokens
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return []


def delete_keys_by_names(
    key_names: List[str],
    api_url: str = "https://aihubmix.com",
    access_token: str = None
) -> Dict[str, int]:
    """
    Batch delete API keys by names
    
    Args:
        key_names: List of key names to delete
        api_url: API base URL
        access_token: Access token
    
    Returns:
        Delete statistics {'success': count, 'failed': count}
    """
    if access_token is None:
        raise ValueError("access_token is required")
    
    headers = {
        "Authorization": access_token,
        "Content-Type": "application/json"
    }
    
    print(f"Deleting {len(key_names)} keys...")
    all_tokens = []
    page = 0
    page_size = 10
    
    while True:
        params = {'p': page, 'size': page_size}
        response = requests.get(f"{api_url}/api/token/", headers=headers, params=params)
        
        if response.status_code != 200:
            return {'success': 0, 'failed': len(key_names)}
        
        data = response.json()
        if not data.get("success", False):
            return {'success': 0, 'failed': len(key_names)}
        
        page_tokens = data.get("data", [])
        if not page_tokens:
            break
        
        all_tokens.extend(page_tokens)
        if len(page_tokens) < page_size:
            break
        
        page += 1
    
    # Map name to ID
    name_to_id = {str(token.get('name', '')): token.get('id') for token in all_tokens}
    
    success_count = 0
    failed_count = 0
    not_found = []
    
    for idx, key_name in enumerate(key_names, 1):
        key_name_str = str(key_name)
        
        if key_name_str not in name_to_id:
            not_found.append(key_name_str)
            failed_count += 1
            continue
        
        key_id = name_to_id[key_name_str]
        
        try:
            response = requests.delete(f"{api_url}/api/token/{key_id}", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success", False):
                    print(f"✓ [{idx}/{len(key_names)}] {key_name_str}")
                    success_count += 1
                else:
                    print(f"✗ [{idx}/{len(key_names)}] {key_name_str}")
                    failed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"\nDeleted: {success_count}/{len(key_names)}")
    if not_found:
        print(f"Not found: {', '.join(not_found)}")
    
    return {'success': success_count, 'failed': failed_count}


def delete_keys_by_range(
    start_key_name: str,
    num_keys: int,
    api_url: str = "https://aihubmix.com",
    access_token: str = None
) -> Dict[str, int]:
    """
    Batch delete API keys by name range
    
    Args:
        start_key_name: Starting key name (numeric string)
        num_keys: Number of keys to delete
        api_url: API base URL
        access_token: Access token
    
    Returns:
        Delete statistics
    """
    try:
        start_num = int(start_key_name)
    except ValueError:
        raise ValueError(f"start_key_name must be numeric string, got: {start_key_name}")
    
    key_names = [str(start_num + i) for i in range(num_keys)]
    
    print(f"Will delete keys from {start_key_name} to {start_num + num_keys - 1}")
    
    confirm = input(f"\nConfirm delete {num_keys} keys? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("Cancelled")
        return {'success': 0, 'failed': 0}
    
    return delete_keys_by_names(key_names, api_url, access_token)


def load_existing_env(env_file: str = ".env") -> Dict[str, str]:
    """
    Load existing .env file content
    
    Args:
        env_file: .env file path
    
    Returns:
        Environment variables dictionary
    """
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


if __name__ == '__main__':
    access_token = "a4e1bf1505b245f8ba52a404a2946dd8"

    # Mode 1: List all available API keys
    # all_keys = list_all_keys(access_token=access_token)
    
    # Mode 2: Generate new API keys with unlimited quota
    # keys = generate_api_keys(
    #     start_key_name="86",
    #     num_keys=24,
    #     access_token=access_token,
    #     unlimited_quota=True,
    #     expired_time=-1,
    # )
    
    # Mode 3: Query existing keys and save to .env
    keys = query_existing_keys(
        start_key_name="86",
        num_keys=100,
        access_token=access_token,
        output_file=".env"
    )
    
    # Mode 4: Delete keys by name range
    # result = delete_keys_by_range(
    #     start_key_name="0",
    #     num_keys=24,
    #     access_token=access_token
    # )