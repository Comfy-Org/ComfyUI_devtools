class ErrorRaiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "raise_error"
    CATEGORY = "DevTools"
    DESCRIPTION = "Raise an error for development purposes"

    def raise_error(self):
        raise Exception("Error node was called!")


NODE_CLASS_MAPPINGS = {
    "DevToolsErrorRaiseNode": ErrorRaiseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DevToolsErrorRaiseNode": "Raise Error",
}
