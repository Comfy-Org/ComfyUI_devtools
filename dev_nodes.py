import torch
import comfy.utils as utils
from comfy.model_patcher import ModelPatcher
import nodes
import time
import os
import folder_paths

import math
import json
from typing import Dict, List, Any
import numpy as np
from PIL import Image


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert IMAGE tensor [B,H,W,C] float32 in [0,1] to PIL.Image (first in batch)."""
    if isinstance(img_tensor, (list, tuple)):
        img_tensor = img_tensor[0]
    if isinstance(img_tensor, torch.Tensor) and img_tensor.ndim == 4:
        img_tensor = img_tensor[0]
    arr = img_tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    # Ensures last axis is channels (H, W, C) that PIL expects
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        pass
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        # Sometimes tensors can be [C, H, W]
        arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def _save_pil(image: Image.Image, prefix: str) -> Dict[str, str]:
    out_dir = folder_paths.get_output_directory()
    subfolder = "devtools"
    _ensure_dir(os.path.join(out_dir, subfolder))
    fname = f"{prefix}_{int(time.time() * 1000)}.png"
    image.save(os.path.join(out_dir, subfolder, fname))
    return {"filename": fname, "subfolder": subfolder, "type": "output"}


def _file_src(meta: Dict[str, str]) -> str:
    sub = meta.get("subfolder", "")
    typ = meta.get("type", "output")
    return f"/view?filename={meta['filename']}&subfolder={sub}&type={typ}"

#####################################
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


class NodeWithOptionalComboInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "optional_combo_input": (
                    [f"Random Unique Option {time.time()}" for _ in range(8)],
                    {"default": None},
                )
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "node_with_optional_combo_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with an optional combo input that returns unique values every time INPUT_TYPES is called"

    def node_with_optional_combo_input(self, optional_combo_input=None):
        print(f"Calling node with optional_combo_input: {optional_combo_input}")
        return (optional_combo_input,)


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


class NodeWithDefaultInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int_input": ("INT", {"defaultInput": True}),
                "int_input_widget": ("INT", {"default": 1}),
            },
            "optional": {"float_input": ("FLOAT", {"defaultInput": True})},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "node_with_default_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a default input"

    def node_with_default_input(
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
        return {"ui": {"text": string_or_int_input}}


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
            "optional": {"dummy_float": ("FLOAT", {"default": 0.0})},
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

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint"

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
                            "query_params": {"sort": "true"},
                        }
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint with query params"

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
                            "max_retries": 10,
                            "timeout": 256,
                        }
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that fetches options from a remote endpoint and refreshes them periodically"

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class RemoteWidgetNodeWithRefreshButton:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                            "refresh_button": True,
                        },
                    },
                ),
            },
        }

    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that lazily fetches options from a remote endpoint and has a refresh button to manually reload options"
    RETURN_TYPES = ("STRING",)

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class RemoteWidgetNodeWithControlAfterRefresh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_widget_value": (
                    "COMBO",
                    {
                        "remote": {
                            "route": "/api/models/checkpoints",
                            "refresh_button": True,
                            "control_after_refresh": "first",
                        }
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remote_widget"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that fetches options and selects the first option after a manual refresh"

    def remote_widget(self, remote_widget_value: str):
        return (remote_widget_value,)


class NodeWithOutputCombo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subset_options": (["A", "B"], {"forceInput": True}),
                "subset_options_v2": ("COMBO", {"options": ["A", "B"], "forceInput": True}),
            }
        }

    RETURN_TYPES = (["A", "B", "C"],)
    FUNCTION = "node_with_output_combo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that outputs a combo type"

    def node_with_output_combo(self, subset_options: str, subset_options_v2: str):
        return (subset_options_v2 or subset_options)


class MultiSelectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foo": (
                    "COMBO",
                    {
                        "options": ["A", "B", "C"],
                        "multi_select": {
                            "placeholder": "Choose foos",
                            "chip": True,
                        },
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "multi_select_node"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that outputs a multi select type"

    def multi_select_node(self, foo: list[str]) -> list[str]:
        return (foo,)


class LoadAnimatedImageTest(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".webp")
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {"image": (sorted(files), {"animated_image_upload": True})},
        }


class NodeWithValidation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"int_input": ("INT",)},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, int_input: int):
        if int_input < 0:
            raise ValueError("int_input must be greater than 0")
        return True

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that validates an input"
    OUTPUT_NODE = True

    def execute(self, int_input: int):
        print(f"int_input: {int_input}")
        return tuple()

class NodeWithV2ComboInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"combo_input": ("COMBO", {"options": ["A", "B"]})}}

    RETURN_TYPES = ("COMBO",)
    FUNCTION = "node_with_v2_combo_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that outputs a combo type that adheres to the v2 combo input spec"

    def node_with_v2_combo_input(self, combo_input: str):
        return (combo_input,)

# 1) Button (input)
class DevToolsButtonWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"button": ("BUTTON", {"label": "Click Me"})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "on_click"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that shows a clickable Button widget"

    def on_click(self, button=None):
        return tuple()


# 2) InputText – single text field
class DevToolsInputTextWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "placeholder": "Enter text..."})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes an Input Text widget"

    def echo(self, text: str):
        return (text,)


# 3) Select – simple options
class DevToolsSelectWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"option": ("COMBO", {"options": ["Option 1", "Option 2", "Option 3"]})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "select"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a Select widget"

    def select(self, option: str):
        return (option,)


# 4) ColorPicker – single color
class DevToolsColorPickerWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"color": ("COLOR", {"default": "#22c55e"})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a Color Picker widget"

    def echo(self, color: str):
        return (color,)


# 5) MultiSelect – checkbox list with chips
class DevToolsMultiSelectWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": (
                    "COMBO",
                    {
                        "options": ["Option 1", "Option 2", "Option 3"],
                        "multi_select": {"placeholder": "Select options...", "chip": True},
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a Multi Select widget"

    def echo(self, values: List[str]):
        return (values or [],)


# 6) SelectButton – three pill options
class DevToolsSelectButtonWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"size": ("SELECTBUTTON", {"options": ["Small", "Medium", "Large"], "default": "Medium"})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a Select Button widget"

    def echo(self, size: str):
        return (size,)


# 7) Slider – fine decimals
class DevToolsSliderWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("FLOAT", {"display": "slider", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001})}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a Slider widget"

    def echo(self, value: float):
        return (value,)


# 7b) INT Slider – integer steps
class DevToolsIntSliderWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"display": "slider", "default": 5, "min": 0, "max": 10, "step": 1})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes an INT Slider widget"

    def echo(self, value: int):
        return (value,)


# 8) Textarea – single field
class DevToolsTextareaWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("TEXTAREA", {"default": "This is a sample text in the textarea component.", "rows": 5, "cols": 40})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a Textarea widget"

    def echo(self, text: str):
        return (text,)


# 9) ToggleSwitch – single boolean
class DevToolsToggleSwitchWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"toggle": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a Toggle Switch widget"

    def echo(self, toggle: bool):
        return (toggle,)


# 10) Chart (preview)
class DevToolsChartWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"chart_type": ("COMBO", {"options": ["line", "bar", "scatter", "radar", "pie"], "default": "bar"})},
            "optional": {"data_json": ("TEXTAREA", {"default": "", "rows": 8, "cols": 60})},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that previews a Chart widget with multiple chart types"

    def _default_data(self, chart_type: str):
        points = 12
        labels = list(range(points))
        if chart_type == "pie":
            return {"labels": ["A", "B", "C", "D", "E"], "datasets": [{"label": "Pie", "data": [12, 19, 3, 5, 2]}]}
        if chart_type == "scatter":
            d1 = [{"x": i, "y": math.sin(i / 2)} for i in labels]
            d2 = [{"x": i, "y": math.cos(i / 3)} for i in labels]
            return {"datasets": [{"label": "Sine", "data": d1}, {"label": "Cos", "data": d2}]}
        # line/bar/radar
        y1 = [math.sin(i / 2) for i in labels]
        y2 = [math.cos(i / 3) for i in labels]
        return {"labels": labels, "datasets": [{"label": "Sine", "data": y1}, {"label": "Cos", "data": y2}]}

    def render(self, chart_type: str, data_json: str = ""):
        data = None
        if data_json:
            try:
                parsed = json.loads(data_json)
                if isinstance(parsed, dict) and ("datasets" in parsed or "labels" in parsed):
                    data = parsed
            except Exception:
                data = None
        if data is None:
            data = self._default_data(chart_type)

        return {
            "ui": {
                "widgets": [
                    {
                        "type": "CHART",
                        "chartType": chart_type,
                        "title": "Chart",
                        "data": data,
                        "options": {"animation": False},
                    }
                ]
            }
        }


# 11) Image (preview)
class DevToolsImageWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that previews an Image widget"

    def render(self, image: torch.Tensor):
        meta = _save_pil(_tensor_to_pil(image), "image")
        return {"ui": {"widgets": [{"type": "IMAGE", "src": _file_src(meta), "image": meta}]}}


# 12) ImageCompare (preview) – slider
class DevToolsImageCompareWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"left_image": ("IMAGE",), "right_image": ("IMAGE",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "compare"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that previews an Image Compare widget"

    def compare(self, left_image: torch.Tensor, right_image: torch.Tensor):
        l = _save_pil(_tensor_to_pil(left_image), "left")
        r = _save_pil(_tensor_to_pil(right_image), "right")
        return {
            "ui": {
                "widgets": [
                    {"type": "IMAGECOMPARE", "mode": "slider", "left": {"image": l}, "right": {"image": r}}
                ]
            }
        }


# 12b) ImageCompare (preview) – side-by-side
class DevToolsImageCompareSideBySideWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"left_image": ("IMAGE",), "right_image": ("IMAGE",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "compare"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that previews an Image Compare widget (side by side)"

    def compare(self, left_image: torch.Tensor, right_image: torch.Tensor):
        l = _save_pil(_tensor_to_pil(left_image), "left")
        r = _save_pil(_tensor_to_pil(right_image), "right")
        return {
            "ui": {
                "widgets": [
                    {"type": "IMAGECOMPARE", "mode": "side-by-side", "left": {"image": l}, "right": {"image": r}}
                ]
            }
        }


# 13) Galleria (input)
class DevToolsGalleriaWidget:
    # Samples are from the PrineVue Component Playground
    _ITEMS = [
        {
            "itemImageSrc": "https://picsum.photos/800/520?random=10",
            "thumbnailImageSrc": "https://picsum.photos/160/100?random=10",
            "alt": "Random placeholder image 1",
            "title": "Placeholder Image 1",
        },
        {
            "itemImageSrc": "https://picsum.photos/800/520?random=11",
            "thumbnailImageSrc": "https://picsum.photos/160/100?random=11",
            "alt": "Random placeholder image 2",
            "title": "Placeholder Image 2",
        },
        {
            "itemImageSrc": "https://picsum.photos/800/520?random=12",
            "thumbnailImageSrc": "https://picsum.photos/160/100?random=12",
            "alt": "Random placeholder image 3",
            "title": "Placeholder Image 3",
        },
        {
            "itemImageSrc": "https://picsum.photos/800/520?random=13",
            "thumbnailImageSrc": "https://picsum.photos/160/100?random=13",
            "alt": "Random placeholder image 4",
            "title": "Placeholder Image 4",
        },
    ]

    @classmethod
    def INPUT_TYPES(cls):
        # Using GALLERIA as an input widget makes it render without running the graph.
        return {"required": {"gallery": ("GALLERIA", {"items": cls._ITEMS})}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that previews a Galleria widget"

    def noop(self, gallery=None):
        return tuple()


# 14) FileUpload – basic (multi)
class DevToolsFileUploadWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"files": ("FILEUPLOAD", {"name": "fileupload", "multiple": True})}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a File Upload widget"

    def noop(self, files=None):
        return tuple()


# 14b) FileUpload – single
class DevToolsFileUploadSingleWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file": ("FILEUPLOAD", {"name": "fileupload_single", "multiple": False})}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a single-file Upload widget"

    def noop(self, file=None):
        return tuple()


# 15) TreeSelect – single
class DevToolsTreeSelectWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "TREESELECT",
                    {
                        "options": [
                            {"label": "Documents", "value": "docs", "children": [
                                {"label": "Work", "value": "work", "children": [
                                    {"label": "Expenses.doc", "value": "expenses"},
                                    {"label": "Resume.doc", "value": "resume"},
                                ]},
                                {"label": "Home", "value": "home", "children": [
                                    {"label": "Invoices.txt", "value": "invoices"}
                                ]},
                            ]},
                            {"label": "Events", "value": "events", "children": [
                                {"label": "Meeting", "value": "meeting"}
                            ]},
                        ],
                        "selectionMode": "single",
                        "placeholder": "Select Item",
                        "showClear": True,
                        "filter": True,
                        "dataKey": "value",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a Tree Select widget"

    def echo(self, value):
        # Normalize whatever the widget returns into a simple string for stability.
        if isinstance(value, dict):
            out = value.get("value") or value.get("label") or json.dumps(value)
        elif isinstance(value, list):
            parts = []
            for v in value:
                if isinstance(v, dict):
                    parts.append(v.get("value") or v.get("label") or "")
                else:
                    parts.append(str(v))
            out = ",".join(parts)
        elif value is None:
            out = ""
        else:
            out = str(value)
        return (out,)


# 15b) TreeSelect – multiple
class DevToolsTreeSelectMultiWidget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": (
                    "TREESELECT",
                    {
                        "options": [
                            {"label": "Fruits", "value": "fruits", "children": [
                                {"label": "Apple", "value": "apple"},
                                {"label": "Banana", "value": "banana"},
                                {"label": "Cherry", "value": "cherry"},
                            ]},
                            {"label": "Vegetables", "value": "veggies", "children": [
                                {"label": "Carrot", "value": "carrot"},
                                {"label": "Lettuce", "value": "lettuce"},
                            ]},
                        ],
                        "selectionMode": "multiple",
                        "placeholder": "Select Items",
                        "showClear": True,
                        "filter": True,
                        "dataKey": "value",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "echo"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes a multi-select Tree Select widget"

    def echo(self, values):
        if isinstance(values, list):
            out = []
            for v in values:
                if isinstance(v, dict):
                    out.append(v.get("value") or v.get("label") or "")
                else:
                    out.append(str(v))
            return (out,)
        return ([],)


# 16) Disabled-state sampler (covers disabled prop rendering)
class DevToolsDisabledWidgets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disabled_text": ("STRING", {"default": "can't edit", "disabled": True}),
                "disabled_combo": ("COMBO", {"options": ["A", "B"], "disabled": True}),
                "disabled_float": ("FLOAT", {"display": "slider", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1, "disabled": True}),
                "disabled_bool": ("BOOLEAN", {"default": False, "disabled": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that exposes disabled widgets for UI state tests"

    def noop(self, **kwargs):
        return tuple()

NODE_CLASS_MAPPINGS = {
    "DevToolsErrorRaiseNode": ErrorRaiseNode,
    "DevToolsErrorRaiseNodeWithMessage": ErrorRaiseNodeWithMessage,
    "DevToolsExperimentalNode": ExperimentalNode,
    "DevToolsDeprecatedNode": DeprecatedNode,
    "DevToolsLongComboDropdown": LongComboDropdown,
    "DevToolsNodeWithOptionalInput": NodeWithOptionalInput,
    "DevToolsNodeWithOptionalComboInput": NodeWithOptionalComboInput,
    "DevToolsNodeWithOnlyOptionalInput": NodeWithOnlyOptionalInput,
    "DevToolsNodeWithOutputList": NodeWithOutputList,
    "DevToolsNodeWithForceInput": NodeWithForceInput,
    "DevToolsNodeWithDefaultInput": NodeWithDefaultInput,
    "DevToolsNodeWithStringInput": NodeWithStringInput,
    "DevToolsNodeWithUnionInput": NodeWithUnionInput,
    "DevToolsSimpleSlider": SimpleSlider,
    "DevToolsNodeWithSeedInput": NodeWithSeedInput,
    "DevToolsObjectPatchNode": ObjectPatchNode,
    "DevToolsNodeWithBooleanInput": NodeWithBooleanInput,
    "DevToolsRemoteWidgetNode": RemoteWidgetNode,
    "DevToolsRemoteWidgetNodeWithParams": RemoteWidgetNodeWithParams,
    "DevToolsRemoteWidgetNodeWithRefresh": RemoteWidgetNodeWithRefresh,
    "DevToolsRemoteWidgetNodeWithRefreshButton": RemoteWidgetNodeWithRefreshButton,
    "DevToolsRemoteWidgetNodeWithControlAfterRefresh": RemoteWidgetNodeWithControlAfterRefresh,
    "DevToolsNodeWithOutputCombo": NodeWithOutputCombo,
    "DevToolsMultiSelectNode": MultiSelectNode,
    "DevToolsLoadAnimatedImageTest": LoadAnimatedImageTest,
    "DevToolsNodeWithValidation": NodeWithValidation,
    "DevToolsNodeWithV2ComboInput": NodeWithV2ComboInput,

    "DevToolsButtonWidget": DevToolsButtonWidget,
    "DevToolsInputTextWidget": DevToolsInputTextWidget,
    "DevToolsSelectWidget": DevToolsSelectWidget,
    "DevToolsColorPickerWidget": DevToolsColorPickerWidget,
    "DevToolsMultiSelectWidget": DevToolsMultiSelectWidget,
    "DevToolsSelectButtonWidget": DevToolsSelectButtonWidget,
    "DevToolsSliderWidget": DevToolsSliderWidget,
    "DevToolsIntSliderWidget": DevToolsIntSliderWidget,
    "DevToolsTextareaWidget": DevToolsTextareaWidget,
    "DevToolsToggleSwitchWidget": DevToolsToggleSwitchWidget,
    "DevToolsChartWidget": DevToolsChartWidget,
    "DevToolsImageWidget": DevToolsImageWidget,
    "DevToolsImageCompareWidget": DevToolsImageCompareWidget,
    "DevToolsImageCompareSideBySideWidget": DevToolsImageCompareSideBySideWidget,
    "DevToolsGalleriaWidget": DevToolsGalleriaWidget,
    "DevToolsFileUploadWidget": DevToolsFileUploadWidget,
    "DevToolsFileUploadSingleWidget": DevToolsFileUploadSingleWidget,
    "DevToolsTreeSelectWidget": DevToolsTreeSelectWidget,
    "DevToolsTreeSelectMultiWidget": DevToolsTreeSelectMultiWidget,
    "DevToolsDisabledWidgets": DevToolsDisabledWidgets,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DevToolsErrorRaiseNode": "Raise Error",
    "DevToolsErrorRaiseNodeWithMessage": "Raise Error with Message",
    "DevToolsExperimentalNode": "Experimental Node",
    "DevToolsDeprecatedNode": "Deprecated Node",
    "DevToolsLongComboDropdown": "Long Combo Dropdown",
    "DevToolsNodeWithOptionalInput": "Node With Optional Input",
    "DevToolsNodeWithOptionalComboInput": "Node With Optional Combo Input",
    "DevToolsNodeWithOnlyOptionalInput": "Node With Only Optional Input",
    "DevToolsNodeWithOutputList": "Node With Output List",
    "DevToolsNodeWithForceInput": "Node With Force Input",
    "DevToolsNodeWithDefaultInput": "Node With Default Input",
    "DevToolsNodeWithStringInput": "Node With String Input",
    "DevToolsNodeWithUnionInput": "Node With Union Input",
    "DevToolsSimpleSlider": "Simple Slider",
    "DevToolsNodeWithSeedInput": "Node With Seed Input",
    "DevToolsObjectPatchNode": "Object Patch Node",
    "DevToolsNodeWithBooleanInput": "Node With Boolean Input",
    "DevToolsRemoteWidgetNode": "Remote Widget Node",
    "DevToolsRemoteWidgetNodeWithParams": "Remote Widget Node With Sort Query Param",
    "DevToolsRemoteWidgetNodeWithRefresh": "Remote Widget Node With 300ms Refresh",
    "DevToolsRemoteWidgetNodeWithRefreshButton": "Remote Widget Node With Refresh Button",
    "DevToolsRemoteWidgetNodeWithControlAfterRefresh": "Remote Widget Node With Refresh Button and Control After Refresh",
    "DevToolsNodeWithOutputCombo": "Node With Output Combo",
    "DevToolsMultiSelectNode": "Multi Select Node",
    "DevToolsLoadAnimatedImageTest": "Load Animated Image",
    "DevToolsNodeWithValidation": "Node With Validation",
    "DevToolsNodeWithV2ComboInput": "Node With V2 Combo Input",

    "DevToolsButtonWidget": "Button",
    "DevToolsInputTextWidget": "Input Text",
    "DevToolsSelectWidget": "Select",
    "DevToolsColorPickerWidget": "Color Picker",
    "DevToolsMultiSelectWidget": "Multi Select",
    "DevToolsSelectButtonWidget": "Select Button",
    "DevToolsSliderWidget": "Slider",
    "DevToolsIntSliderWidget": "Int Slider",
    "DevToolsTextareaWidget": "Textarea",
    "DevToolsToggleSwitchWidget": "Toggle Switch",
    "DevToolsChartWidget": "Chart",
    "DevToolsImageWidget": "Image",
    "DevToolsImageCompareWidget": "Image Compare (Slider)",
    "DevToolsImageCompareSideBySideWidget": "Image Compare (Side by Side)",
    "DevToolsGalleriaWidget": "Galleria",
    "DevToolsFileUploadWidget": "File Upload (Multi)",
    "DevToolsFileUploadSingleWidget": "File Upload (Single)",
    "DevToolsTreeSelectWidget": "Tree Select",
    "DevToolsTreeSelectMultiWidget": "Tree Select (Multi)",
    "DevToolsDisabledWidgets": "Disabled Widgets",
}
