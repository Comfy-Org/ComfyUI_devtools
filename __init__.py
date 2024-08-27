from .dev_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

import os
import server
from aiohttp import web

@server.PromptServer.instance.routes.get("/devtools/fake_model.safetensors")
async def fake_model(request):
    file_path = os.path.join(os.path.dirname(__file__), "fake_model.safetensors")
    return web.FileResponse(file_path)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
