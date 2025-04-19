from datetime import datetime

from pydantic import BaseModel


class HealthCheck(BaseModel):
    """Модель для результатов проверки работоспособности"""

    health: bool
    datetime: datetime
