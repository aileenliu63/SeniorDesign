# config.py

# Private dictionary holding all shared settings
_settings = {
    "windows": {},
    "threshold": 3,
}

def get(key, default=None):
    return _settings.get(key, default)

def set(key, value):
    _settings[key] = value

def all():
    #return entire settings dictionary
    return _settings.copy()
