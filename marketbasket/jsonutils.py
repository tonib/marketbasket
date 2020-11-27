
# In a single file to avoid ImportError: cannot import name '_read_setting' from partially initialized module 'marketbasket.settings' 
# (most likely due to a circular import)

def read_setting( settings_json: dict, key: str, value_type: type, default_value: object) -> object:
    if key in settings_json:
        return value_type( settings_json[key] )
    if isinstance(default_value, Exception):
        raise default_value
    return default_value