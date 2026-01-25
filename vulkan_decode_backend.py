from __future__ import annotations

"""
Vulkan decode backend (Stage 1–2): GPU low-pass decode via vkFFT + DASHI mask on GPU.

This module keeps per-grid Vulkan handles, pipelines, and host-visible buffers so the
decode path can run without CuPy/Torch. Residual synthesis remains CPU-side for now.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import sys
import struct

import numpy as np

CORE_ROOT = Path(__file__).resolve().parent / "dashiCORE"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from gpu_common_methods import compile_shader  # type: ignore
from gpu_vkfft_adapter import VkFFTExecutor  # type: ignore
from gpu_vulkan_dispatcher import (  # type: ignore
    HOST_VISIBLE_COHERENT,
    VulkanHandles,
    _create_buffer,
    _read_buffer,
    create_vulkan_handles,
)

try:
    import vulkan as vk  # type: ignore
except Exception as exc:  # pragma: no cover - only hit when Vulkan is missing
    vk = None  # type: ignore
    _VK_IMPORT_ERROR = exc
else:
    _VK_IMPORT_ERROR = None


@dataclass
class _Pipeline:
    name: str
    shader_path: Path
    spv_path: Path
    descriptor_set_layout: object
    pipeline_layout: object
    pipeline: object
    push_size: int


class VulkanDecodeBackend:
    def __init__(
        self,
        N: int,
        *,
        smooth_k: int,
        majority_iters: int = 8,
        fft_backend: str = "vkfft-vulkan",
    ):
        if vk is None:
            raise RuntimeError(f"vulkan python package not available: {_VK_IMPORT_ERROR}")

        self.N = int(N)
        self.total = int(N * N)
        self.smooth_k = int(smooth_k)
        self.majority_iters = int(majority_iters)
        self.handles: VulkanHandles = create_vulkan_handles()
        self.command_pool = self._create_command_pool()
        self.fft_exec = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend)

        self._pipelines: Dict[str, _Pipeline] = {}
        self._build_pipelines()

        self._buffers = {}
        self._alloc_buffers()

    # --------------------- setup ---------------------
    def _create_command_pool(self):
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.handles.queue_family_index,
        )
        return vk.vkCreateCommandPool(self.handles.device, pool_info, None)

    def _alloc_buffers(self):
        device = self.handles.device
        mem_props = self.handles.mem_props
        usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT

        def _buf(name: str, nbytes: int):
            buf, mem = _create_buffer(device, mem_props, nbytes, usage, HOST_VISIBLE_COHERENT)
            self._buffers[name] = (buf, mem, nbytes)

        real_bytes = self.total * 4  # float32
        int_bytes = self.total * 4  # int32 / uint32
        _buf("omega_lp", real_bytes)
        _buf("tmp", real_bytes)
        _buf("base", real_bytes)
        _buf("sign_a", int_bytes)
        _buf("sign_b", int_bytes)

        self.partial_len = (self.total + 255) // 256
        _buf("partial_max", self.partial_len * 4)

    def _build_pipelines(self):
        shaders = [
            ("c2r", "decode_complex_to_real.comp", 8),
            ("smooth_x", "decode_smooth_x.comp", 12),
            ("smooth_y", "decode_smooth_y.comp", 12),
            ("absmax", "decode_absmax_reduce.comp", 4),
            ("threshold", "decode_threshold.comp", 16),
            ("majority", "decode_majority3x3.comp", 4),
        ]
        root = Path(__file__).resolve().parent / "dashiCORE" / "gpu_shaders"
        for name, fname, push_size in shaders:
            shader_path = root / fname
            spv_path = shader_path.with_suffix(".spv")
            compile_shader(shader_path, spv_path)
            pipeline = self._make_pipeline(name, shader_path, spv_path, push_size, self._binding_count_for(name))
            self._pipelines[name] = pipeline

    def _binding_count_for(self, name: str) -> int:
        return {
            "c2r": 2,
            "smooth_x": 2,
            "smooth_y": 2,
            "absmax": 3,
            "threshold": 3,
            "majority": 2,
        }[name]

    def _make_pipeline(self, name: str, shader_path: Path, spv_path: Path, push_size: int, bindings: int) -> _Pipeline:
        device = self.handles.device
        # Descriptor set layout
        binding_layouts = []
        for b in range(bindings):
            binding_layouts.append(
                vk.VkDescriptorSetLayoutBinding(
                    binding=b,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    pImmutableSamplers=None,
                )
            )
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(binding_layouts),
            pBindings=binding_layouts,
        )
        descriptor_set_layout = vk.vkCreateDescriptorSetLayout(device, layout_info, None)

        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=push_size,
        )
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_range],
        )
        pipeline_layout = vk.vkCreatePipelineLayout(device, pipeline_layout_info, None)

        code_bytes = spv_path.read_bytes()
        shader_module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code_bytes),
            pCode=code_bytes,
        )
        shader_module = vk.vkCreateShaderModule(device, shader_module_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=pipeline_layout,
        )
        pipeline = vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)[0]

        return _Pipeline(
            name=name,
            shader_path=shader_path,
            spv_path=spv_path,
            descriptor_set_layout=descriptor_set_layout,
            pipeline_layout=pipeline_layout,
            pipeline=pipeline,
            push_size=push_size,
        )

    # --------------------- helpers ---------------------
    def _alloc_command_buffer(self):
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vk.vkAllocateCommandBuffers(self.handles.device, alloc_info)[0]

    def _submit_and_wait(self, cmd):
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vk.vkCreateFence(self.handles.device, fence_info, None)
        vk.vkQueueSubmit(self.handles.queue, 1, [submit_info], fence)
        vk.vkWaitForFences(self.handles.device, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        vk.vkDestroyFence(self.handles.device, fence, None)

    def _allocate_descriptor_set(self, pipeline: _Pipeline, buffers: Tuple[Tuple[object, int], ...]):
        device = self.handles.device
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=len(buffers),
            )
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
            maxSets=1,
        )
        descriptor_pool = vk.vkCreateDescriptorPool(device, pool_info, None)
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[pipeline.descriptor_set_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(device, alloc_info)[0]

        writes = []
        for binding, (buf, nbytes) in enumerate(buffers):
            info = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=nbytes)
            writes.append(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=binding,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[info],
                )
            )
        vk.vkUpdateDescriptorSets(device, len(writes), writes, 0, None)
        return descriptor_pool, descriptor_set

    def _dispatch(self, name: str, buffers: Tuple[Tuple[object, int], ...], push_bytes: bytes, groups: Tuple[int, int, int]):
        pipeline = self._pipelines[name]
        device = self.handles.device
        cmd = self._alloc_command_buffer()

        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        descriptor_pool, descriptor_set = self._allocate_descriptor_set(pipeline, buffers)

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout,
            0,
            1,
            [descriptor_set],
            0,
            None,
        )
        if push_bytes:
            vk.vkCmdPushConstants(
                cmd,
                pipeline.pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_bytes),
                push_bytes,
            )
        gx, gy, gz = groups
        vk.vkCmdDispatch(cmd, gx, gy, gz)
        vk.vkEndCommandBuffer(cmd)

        self._submit_and_wait(cmd)
        vk.vkDestroyDescriptorPool(device, descriptor_pool, None)
        vk.vkFreeCommandBuffers(device, self.command_pool, 1, [cmd])

    # --------------------- public API ---------------------
    def decode_lowpass_mask(self, oh: np.ndarray, tau: float, smooth_k: int | None = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Return (omega_lp, sign_mask, timings)."""
        if oh.dtype != np.complex64:
            oh = oh.astype(np.complex64, copy=False)
        k = int(smooth_k) if smooth_k is not None else self.smooth_k

        timings: Dict[str, float] = {}

        # vkFFT IFFT in-place to device buffer
        t0 = time.perf_counter()
        plan = self.fft_exec._get_plan(oh, direction="ifft")  # type: ignore[attr-defined]
        if plan is None:
            raise RuntimeError("vkFFT plan unavailable for Vulkan decode")
        self.fft_exec._upload(oh, plan)  # type: ignore[attr-defined]
        self.fft_exec._run_vkfft(plan, inverse=True)  # type: ignore[attr-defined]
        vk.vkQueueWaitIdle(self.handles.queue)
        timings["ifft_lp_ms"] = 1000 * (time.perf_counter() - t0)

        N = self.N
        total = self.total

        # Complex -> real (omega_lp)
        scale = np.float32(1.0 / float(N * N))
        gx = (N + 15) // 16
        gy = (N + 15) // 16
        t1 = time.perf_counter()
        self._dispatch(
            "c2r",
            buffers=(
                (plan.device_buffer, oh.nbytes),  # type: ignore[attr-defined]
                (self._buffers["omega_lp"][0], self._buffers["omega_lp"][2]),
            ),
            push_bytes=struct.pack("<If", N, float(scale)),
            groups=(gx, gy, 1),
        )
        timings["complex_to_real_ms"] = 1000 * (time.perf_counter() - t1)

        inv_k = np.float32(1.0 / float(max(1, k)))
        # smooth x
        t2 = time.perf_counter()
        push = struct.pack("<IIf", N, k, float(inv_k))
        self._dispatch(
            "smooth_x",
            buffers=(
                (self._buffers["omega_lp"][0], self._buffers["omega_lp"][2]),
                (self._buffers["tmp"][0], self._buffers["tmp"][2]),
            ),
            push_bytes=push,
            groups=(gx, gy, 1),
        )
        # smooth y (tmp -> base)
        self._dispatch(
            "smooth_y",
            buffers=(
                (self._buffers["tmp"][0], self._buffers["tmp"][2]),
                (self._buffers["base"][0], self._buffers["base"][2]),
            ),
            push_bytes=push,
            groups=(gx, gy, 1),
        )
        timings["smooth_ms"] = 1000 * (time.perf_counter() - t2)

        # absmax reduce into partial_max
        t3 = time.perf_counter()
        g_abs = (self.partial_len, 1, 1)
        push_abs = struct.pack("<I", total)
        self._dispatch(
            "absmax",
            buffers=(
                (self._buffers["omega_lp"][0], self._buffers["omega_lp"][2]),
                (self._buffers["base"][0], self._buffers["base"][2]),
                (self._buffers["partial_max"][0], self._buffers["partial_max"][2]),
            ),
            push_bytes=push_abs,
            groups=g_abs,
        )
        partial = _read_buffer(self.handles.device, self._buffers["partial_max"][1], (self.partial_len,), np.float32)
        max_val = float(np.max(partial))
        timings["absmax_ms"] = 1000 * (time.perf_counter() - t3)

        inv_max = np.float32(1.0 / max(max_val, 1e-12))
        t4 = time.perf_counter()
        push_thr = struct.pack("<Ifff", total, float(tau), float(inv_max), 1e-12)
        g_thr = ((total + 255) // 256, 1, 1)
        self._dispatch(
            "threshold",
            buffers=(
                (self._buffers["omega_lp"][0], self._buffers["omega_lp"][2]),
                (self._buffers["base"][0], self._buffers["base"][2]),
                (self._buffers["sign_a"][0], self._buffers["sign_a"][2]),
            ),
            push_bytes=push_thr,
            groups=g_thr,
        )
        timings["threshold_ms"] = 1000 * (time.perf_counter() - t4)

        # majority iterations
        t5 = time.perf_counter()
        for it in range(self.majority_iters):
            in_name = "sign_a" if it % 2 == 0 else "sign_b"
            out_name = "sign_b" if it % 2 == 0 else "sign_a"
            push_maj = struct.pack("<I", N)
            self._dispatch(
                "majority",
                buffers=(
                    (self._buffers[in_name][0], self._buffers[in_name][2]),
                    (self._buffers[out_name][0], self._buffers[out_name][2]),
                ),
                push_bytes=push_maj,
                groups=(gx, gy, 1),
            )
        timings["majority_ms"] = 1000 * (time.perf_counter() - t5)

        final_sign_name = "sign_b" if (self.majority_iters % 2 == 1) else "sign_a"

        omega_lp = _read_buffer(self.handles.device, self._buffers["omega_lp"][1], (N, N), np.float32)
        sign = _read_buffer(self.handles.device, self._buffers[final_sign_name][1], (N, N), np.int32)
        return omega_lp, sign, timings


_CACHE: Dict[Tuple[int, int, int, str], VulkanDecodeBackend] = {}


def get_vulkan_decoder(N: int, *, smooth_k: int, majority_iters: int = 8, fft_backend: str = "vkfft-vulkan") -> VulkanDecodeBackend:
    key = (N, int(smooth_k), int(majority_iters), fft_backend)
    if key not in _CACHE:
        _CACHE[key] = VulkanDecodeBackend(N, smooth_k=smooth_k, majority_iters=majority_iters, fft_backend=fft_backend)
    return _CACHE[key]
