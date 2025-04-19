import datetime
from multiprocessing import freeze_support

import uvicorn

from settings import settings


def main():
    freeze_support()  # Needed for pyinstaller for multiprocessing on WindowsOS

    # num_workers = int(cpu_count() * 0.75)
    num_workers = 1

    uvicorn.run(
        "app:app",
        host=settings.service.server_host,
        port=settings.service.server_port,
        log_level=settings.logging.level,
        workers=num_workers,
        reload=False,
        use_colors=False,
    )

    now = datetime.datetime.now(tz=datetime.UTC)
    dtString = now.strftime("%d.%m.%Y %H:%M:%S")

    print(f"Server stopped at {dtString}")


if __name__ == "__main__":
    main()
