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


class ErrorRaiseNodeWithMessage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"message": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "raise_error"
    CATEGORY = "DevTools"
    DESCRIPTION = "Raise an error with message for development purposes"

    def raise_error(self, message: str):
        raise Exception(message)


NODE_CLASS_MAPPINGS = {
    "DevToolsErrorRaiseNode": ErrorRaiseNode,
    "DevToolsErrorRaiseNodeWithMessage": ErrorRaiseNodeWithMessage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DevToolsErrorRaiseNode": "Raise Error",
    "DevToolsErrorRaiseNodeWithMessage": "Raise Error with Message",
}
