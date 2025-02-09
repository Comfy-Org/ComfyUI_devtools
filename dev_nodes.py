import torch
import comfy.utils as utils
from comfy.model_patcher import ModelPatcher


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
        return {"required": {"option": ([f"Option {i}" for i in range(1_000)],)}}

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
                "clip": ("CLIP", {}),
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

    def node_with_force_input(
        self, int_input: int, int_input_widget: int, float_input: float = 0.0
    ):
        print(
            f"int_input: {int_input}, int_input_widget: {int_input_widget}, float_input: {float_input}"
        )


class NodeWithStringInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string_input": ("STRING",)}}

    RETURN_TYPES = ()
    FUNCTION = "node_with_string_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a string input"

    def node_with_string_input(self, string_input: str):
        print(f"string_input: {string_input}")


class NodeWithUnionInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "string_or_int_input": ("STRING,INT",),
                "string_input": ("STRING", {"forceInput": True}),
                "int_input": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "node_with_union_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a union input"

    def node_with_union_input(
        self,
        string_or_int_input: str | int = "",
        string_input: str = "",
        int_input: int = 0,
    ):
        print(
            f"string_or_int_input: {string_or_int_input}, string_input: {string_input}, int_input: {int_input}"
        )
        return {
            "ui": {
                "text": string_or_int_input,
            }
        }


class NodeWithBooleanInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"boolean_input": ("BOOLEAN",)}}

    RETURN_TYPES = ()
    FUNCTION = "node_with_boolean_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a boolean input"

    def node_with_boolean_input(self, boolean_input: bool):
        print(f"boolean_input: {boolean_input}")


class SimpleSlider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {
                        "display": "slider",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "DevTools"

    def execute(self, value):
        return (value,)


class NodeWithSeedInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0})}}

    RETURN_TYPES = ()
    FUNCTION = "node_with_seed_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a seed input"
    OUTPUT_NODE = True

    def node_with_seed_input(self, seed: int):
        print(f"seed: {seed}")


class DummyPatch(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, dummy_float: float = 0.0):
        super().__init__()
        self.module = module
        self.dummy_float = dummy_float

    def forward(self, *args, **kwargs):
        if isinstance(self.module, DummyPatch):
            raise Exception(f"Calling nested dummy patch! {self.dummy_float}")

        return self.module(*args, **kwargs)


class ObjectPatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "target_module": ("STRING", {"multiline": True}),
            },
            "optional": {
                "dummy_float": ("FLOAT", {"default": 0.0 }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that applies an object patch"

    def apply_patch(
        self, model: ModelPatcher, target_module: str, dummy_float: float = 0.0
    ) -> ModelPatcher:
        module = utils.get_attr(model.model, target_module)
        work_model = model.clone()
        work_model.add_object_patch(target_module, DummyPatch(module, dummy_float))
        return (work_model,)


class RemoteWidgetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint"
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class RemoteWidgetNodeWithParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                            "query_params": {
                                "sort": "true",
                            },
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = (
        "A node that lazily fetches options from a remote endpoint with query params"
    )
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class RemoteWidgetNodeWithRefresh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                            "refresh": 300,
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint and refresh the options every 300 ms"
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


NODE_CLASS_MAPPINGS = {
    "DevToolsErrorRaiseNode": ErrorRaiseNode,
    "DevToolsErrorRaiseNodeWithMessage": ErrorRaiseNodeWithMessage,
    "DevToolsExperimentalNode": ExperimentalNode,
    "DevToolsDeprecatedNode": DeprecatedNode,
    "DevToolsLongComboDropdown": LongComboDropdown,
    "DevToolsNodeWithOptionalInput": NodeWithOptionalInput,
    "DevToolsNodeWithOnlyOptionalInput": NodeWithOnlyOptionalInput,
    "DevToolsNodeWithOutputList": NodeWithOutputList,
    "DevToolsNodeWithForceInput": NodeWithForceInput,
    "DevToolsNodeWithStringInput": NodeWithStringInput,
    "DevToolsNodeWithUnionInput": NodeWithUnionInput,
    "DevToolsSimpleSlider": SimpleSlider,
    "DevToolsNodeWithSeedInput": NodeWithSeedInput,
    "DevToolsObjectPatchNode": ObjectPatchNode,
    "DevToolsNodeWithBooleanInput": NodeWithBooleanInput,
    "DevToolsRemoteWidgetNode": RemoteWidgetNode,
    "DevToolsRemoteWidgetNodeWithParams": RemoteWidgetNodeWithParams,
    "DevToolsRemoteWidgetNodeWithRefresh": RemoteWidgetNodeWithRefresh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DevToolsErrorRaiseNode": "Raise Error",
    "DevToolsErrorRaiseNodeWithMessage": "Raise Error with Message",
    "DevToolsExperimentalNode": "Experimental Node",
    "DevToolsDeprecatedNode": "Deprecated Node",
    "DevToolsLongComboDropdown": "Long Combo Dropdown",
    "DevToolsNodeWithOptionalInput": "Node With Optional Input",
    "DevToolsNodeWithOnlyOptionalInput": "Node With Only Optional Input",
    "DevToolsNodeWithOutputList": "Node With Output List",
    "DevToolsNodeWithForceInput": "Node With Force Input",
    "DevToolsNodeWithStringInput": "Node With String Input",
    "DevToolsNodeWithUnionInput": "Node With Union Input",
    "DevToolsSimpleSlider": "Simple Slider",
    "DevToolsNodeWithSeedInput": "Node With Seed Input",
    "DevToolsObjectPatchNode": "Object Patch Node",
    "DevToolsNodeWithBooleanInput": "Node With Boolean Input",
    "DevToolsRemoteWidgetNode": "Remote Widget Node",
    "DevToolsRemoteWidgetNodeWithParams": "Remote Widget Node With Sort Query Param",
    "DevToolsRemoteWidgetNodeWithRefresh": "Remote Widget Node With 300ms Refresh",
}
