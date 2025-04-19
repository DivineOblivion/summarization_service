import datetime

from fastapi import APIRouter, status

from settings import settings

from .models import HealthCheck

InfoRouter = APIRouter(tags=["Info"])


@InfoRouter.get(
    "/health",
    summary="Проверка работоспособности сервиса",
    description="Проверяет работоспособность сервиса",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """Проверка работоспособности

    Возвращает:
        HealthCheck: Результат проверки работоспособности
    """
    return HealthCheck(health=True, datetime=datetime.datetime.now(tz=datetime.UTC))


@InfoRouter.get(
    "/version",
    summary="Версия сервиса",
    description="Получение версии сервиса",
    response_description="Версия сервиса",
    status_code=status.HTTP_200_OK,
    response_model=str,
)
def get_version() -> str:
    """Получение версии сервиса

    Возвращает:
        str: Версия сервиса
    """
    return settings.service.version
