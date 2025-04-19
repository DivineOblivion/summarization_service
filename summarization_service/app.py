import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request, templating

from api.info_router.router import InfoRouter
from api.logic_router.router import LogicRouter
from settings import settings
from tools.docs import generate_offline_docs

if sys.platform.startswith("win"):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


app = FastAPI(
    title=settings.service.service_name + " API",
    version=settings.service.version,
    description="",
    docs_url=None,
    redocs_url=None,
)


api_v1_prefix = "/api/v1"

app.include_router(InfoRouter, prefix=api_v1_prefix)
app.include_router(LogicRouter, prefix=api_v1_prefix)


main_path = os.path.split(__file__)[0]
templates = templating.Jinja2Templates(Path(main_path, "templates"))

if settings.service.offline_docs:
    static_path = Path(main_path, "static")
    generate_offline_docs(app, static_path)
else:
    app.docs_url = "/docs"
    app.redoc_url = "/redocs"
    app.setup()


@app.get("/")
def index_page(request: Request):
    return templates.TemplateResponse(request, "index.html")
