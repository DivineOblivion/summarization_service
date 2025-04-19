import logging

from pydantic_settings import BaseSettings


class Service(BaseSettings):
    service_name: str = "Summarization Service"
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    offline_docs: bool = True
    version: str = "v0.1.0"


class Logging(BaseSettings):
    level: int = logging.DEBUG
    keep: int = 8


class Settings(BaseSettings):
    service: Service = Service()
    logging: Logging = Logging()


settings = Settings()
