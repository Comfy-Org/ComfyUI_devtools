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


class ExperimentalNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "experimental_function"
    CATEGORY = "DevTools"
    DESCRIPTION = "A experimental node"

    EXPERIMENTAL = True

    def experimental_function(self):
        print("Experimental node was called!")


class DeprecatedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "deprecated_function"
    CATEGORY = "DevTools"
    DESCRIPTION = "A deprecated node"

    DEPRECATED = True

    def deprecated_function(self):
        print("Deprecated node was called!")


class LongComboDropdown:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"option": ([f"Option {i}" for i in range(1000 * 1000)],)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "long_combo_dropdown"
    CATEGORY = "DevTools"
    DESCRIPTION = "A long combo dropdown"

    def long_combo_dropdown(self, option: str):
        print(option)


class NodeWithOptionalInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"required_input": ("IMAGE",)},
            "optional": {"optional_input": ("IMAGE", {"default": None})},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node_with_optional_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with an optional input"

    def node_with_optional_input(self, required_input, optional_input=None):
        print(
            f"Calling node with required_input: {required_input} and optional_input: {optional_input}"
        )
        return (required_input,)


class NodeWithOnlyOptionalInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}), 
                "clip": ("CLIP", {})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "node_with_only_optional_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with only optional input"

    def node_with_only_optional_input(self, clip=None, text=None):
        pass


class NodeWithOutputList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = (
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "INTEGER OUTPUT",
        "INTEGER LIST OUTPUT",
    )
    OUTPUT_IS_LIST = (
        False,
        True,
    )
    FUNCTION = "node_with_output_list"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with an output list"

    def node_with_output_list(self):
        return (1, [1, 2, 3])


class NodeWithForceInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int_input": ("INT", {"forceInput": True}),
                "int_input_widget": ("INT", {"default": 1}),
            },
            "optional": {"float_input": ("FLOAT", {"forceInput": True})},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "node_with_force_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a forced input"

    def node_with_force_input(self, int_input: int, int_input_widget: int, float_input: float = 0.0):
        print(f"int_input: {int_input}, int_input_widget: {int_input_widget}, float_input: {float_input}")


NODE_CLASS_MAPPINGS = {
    "DevToolsErrorRaiseNode": ErrorRaiseNode,
    "DevToolsErrorRaiseNodeWithMessage": ErrorRaiseNodeWithMessage,
    "DevToolsExperimentalNode": ExperimentalNode,
    "DevToolsDeprecatedNode": DeprecatedNode,
    "DevToolsLongComboDropdown": LongComboDropdown,
    "DevToolsNodeWithOptionalInput": NodeWithOptionalInput,
    "DevToolsNodeWithOutputList": NodeWithOutputList,
    "DevToolsNodeWithForceInput": NodeWithForceInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DevToolsErrorRaiseNode": "Raise Error",
    "DevToolsErrorRaiseNodeWithMessage": "Raise Error with Message",
    "DevToolsExperimentalNode": "Experimental Node",
    "DevToolsDeprecatedNode": "Deprecated Node",
    "DevToolsLongComboDropdown": "Long Combo Dropdown",
    "DevToolsNodeWithOptionalInput": "Node With Optional Input",
    "DevToolsNodeWithOutputList": "Node With Output List",
    "DevToolsNodeWithForceInput": "Node With Force Input",
}
