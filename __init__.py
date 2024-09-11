from .dev_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

import os
import server
from aiohttp import web
from folder_paths import models_dir

@server.PromptServer.instance.routes.get("/devtools/fake_model.safetensors")
async def fake_model(request):
    file_path = os.path.join(os.path.dirname(__file__), "fake_model.safetensors")
    return web.FileResponse(file_path)

@server.PromptServer.instance.routes.get("/devtools/cleanup_fake_model")
async def cleanup_fake_model(request):
    model_folder = request.query.get('model_folder', 'clip')
    model_path = os.path.join(models_dir, model_folder, 'fake_model.safetensors')
    if os.path.exists(model_path):
        os.remove(model_path)
    return web.Response(status=200, text="Fake model cleaned up")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
