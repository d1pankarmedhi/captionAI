import os


class Settings:
    # Configuration
    SECRET_KEY = "secret12234"  # os.getenv("SECRET_KEY")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
