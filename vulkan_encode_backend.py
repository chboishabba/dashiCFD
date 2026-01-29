from __future__ import annotations

"""GPU encode_proxy backend (vkFFT + Vulkan kernels)."""

import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import sys

from dashi_cfd_operator_v4 import make_grid, ProxyConfig

CORE_ROOT = Path(__file__).resolve().parent / "dashiCORE"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from gpu_common_methods import compile_shader, resolve_shader, resolve_spv  # type: ignore
from gpu_vkfft_adapter import VkFFTExecutor  # type: ignore
from gpu_vulkan_dispatcher import (  # type: ignore
    HOST_VISIBLE_COHERENT,
    VulkanHandles,
    _create_buffer,
    _read_buffer,
    _write_buffer,
    create_vulkan_handles,
)

try:
    import vulkan as vk  # type: ignore
except Exception as exc:  # pragma: no cover
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


class VulkanEncodeBackend:
    """GPU encode_proxy (low-k + anchors + residual energies)."""

    def __init__(
        self,
        N: int,
        cfg: ProxyConfig,
        *,
        fft_backend: str = "vkfft-vulkan",
        spectral_truncation: str = "none",
        trunc_alpha: float = 36.0,
        trunc_power: float = 8.0,
        batch_dispatch: bool = False,
        timing_enabled: bool = True,
    ):
        if vk is None:
            raise RuntimeError(f"vulkan python package not available: {_VK_IMPORT_ERROR}")
        self.N = int(N)
        self.total = int(N * N)
        self.cfg = cfg
        self.spectral_truncation = spectral_truncation
        self.trunc_alpha = float(trunc_alpha)
        self.trunc_power = float(trunc_power)
        self.batch_dispatch = bool(batch_dispatch)
        self.timing_enabled = bool(timing_enabled)

        self.handles: VulkanHandles = create_vulkan_handles()
        self.command_pool = self._create_command_pool()
        self.fft = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend, timing_enabled=timing_enabled)

        self._pipelines: Dict[str, _Pipeline] = {}
        self._buffers = {}
        self._descriptor_pool = None
        self._max_descriptor_bindings = 0
        self._alloc_buffers()
        self._init_k_buffers()
        self._init_fft_plan()
        self._build_pipelines()

        self._mask_low_idx = None
        self._anchor_idx = None
        self._lowk_count = 0
        self._anchor_count = 0
        self._n_mid = 0
        self._n_high = 0
        self._timing_active = False
        self._timing_last: Dict[str, float] = {
            "gpu_time_ms": 0.0,
            "gpu_wait_ms": 0.0,
            "fence_wait_ms": 0.0,
            "queue_wait_ms": 0.0,
        }
        self._timestamp_supported = False
        self._timestamp_period = 0.0
        self._timestamp_mask: Optional[int] = None
        self._init_timestamp_support()

    # --------------------- setup ---------------------
    def _create_command_pool(self):
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.handles.queue_family_index,
        )
        return vk.vkCreateCommandPool(self.handles.device, pool_info, None)

    def _timing_reset(self) -> None:
        if not self.timing_enabled:
            self._timing_active = False
            return
        self._timing_last = {
            "gpu_time_ms": 0.0,
            "gpu_wait_ms": 0.0,
            "fence_wait_ms": 0.0,
            "queue_wait_ms": 0.0,
        }
        self._timing_active = True

    def _timing_finish(self) -> None:
        self._timing_active = False

    def get_last_timings(self) -> Dict[str, float]:
        return dict(self._timing_last) if self.timing_enabled else {}

    def _init_timestamp_support(self) -> None:
        if not self.timing_enabled:
            return
        try:
            props = vk.vkGetPhysicalDeviceProperties(self.handles.physical_device)
            self._timestamp_period = float(props.limits.timestampPeriod or 0.0)
            qprops = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.handles.physical_device)
            valid_bits = int(qprops[self.handles.queue_family_index].timestampValidBits)
            if self._timestamp_period > 0.0 and valid_bits > 0:
                self._timestamp_supported = True
                if valid_bits < 64:
                    self._timestamp_mask = (1 << valid_bits) - 1
        except Exception:
            self._timestamp_supported = False
            self._timestamp_period = 0.0
            self._timestamp_mask = None

    def _create_timestamp_query_pool(self):
        if not (self.timing_enabled and self._timestamp_supported):
            return None
        info = vk.VkQueryPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            queryType=vk.VK_QUERY_TYPE_TIMESTAMP,
            queryCount=2,
        )
        return vk.vkCreateQueryPool(self.handles.device, info, None)

    def _cmd_write_timestamps_begin(self, cmd, query_pool) -> None:
        if query_pool is None:
            return
        if hasattr(vk, "vkCmdResetQueryPool"):
            vk.vkCmdResetQueryPool(cmd, query_pool, 0, 2)
        vk.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0)

    def _cmd_write_timestamps_end(self, cmd, query_pool) -> None:
        if query_pool is None:
            return
        vk.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1)

    def _read_timestamp_ms(self, query_pool) -> float:
        if query_pool is None or not self._timestamp_supported:
            return 0.0
        data_size = 16
        stride = 8
        flags = vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT
        if hasattr(vk, "ffi"):
            data = vk.ffi.new("uint64_t[]", 2)
            vk.vkGetQueryPoolResults(
                self.handles.device,
                query_pool,
                0,
                2,
                data_size,
                data,
                stride,
                flags,
            )
            t0, t1 = int(data[0]), int(data[1])
        else:
            import ctypes

            data = (ctypes.c_uint64 * 2)()
            vk.vkGetQueryPoolResults(
                self.handles.device,
                query_pool,
                0,
                2,
                data_size,
                data,
                stride,
                flags,
            )
            t0, t1 = int(data[0]), int(data[1])
        if self._timestamp_mask is not None:
            mask = self._timestamp_mask
            t0 &= mask
            t1 &= mask
            delta = (t1 - t0) & mask
        else:
            delta = t1 - t0
        return (delta * self._timestamp_period) / 1.0e6

    def _alloc_buffers(self):
        device = self.handles.device
        mem_props = self.handles.mem_props
        usage = (
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )

        def _buf(name: str, nbytes: int):
            buf, mem = _create_buffer(device, mem_props, nbytes, usage, HOST_VISIBLE_COHERENT)
            self._buffers[name] = (buf, mem, nbytes)

        real_bytes = self.total * 4
        _buf("omega", real_bytes)
        _buf("base_a", real_bytes)
        _buf("base_b", real_bytes)
        _buf("residual", real_bytes)
        _buf("mid_energy", real_bytes)
        _buf("high_energy", real_bytes)

        # k-space arrays (float)
        _buf("kx", real_bytes)
        _buf("ky", real_bytes)

        # dynamic index/compact buffers (allocated when set_mask_low/set_anchor_idx called)
        self._buffers["lowk_idx"] = None
        self._buffers["anchor_idx"] = None
        self._buffers["lowk_vals"] = None
        self._buffers["kept_vals"] = None
        self._buffers["mid_idx"] = None
        self._buffers["mid_scores"] = None
        self._buffers["max_partials"] = None
        self._buffers["max_best"] = None

        # reduction buffers
        self.partial_len = (self.total + 255) // 256
        _buf("partial0", self.partial_len * 4)
        _buf("partial1", self.partial_len * 4)
        _buf("scalar", 4)

    def _init_k_buffers(self):
        dx, KX, KY, _ = make_grid(self.N)
        self.dx = float(dx)
        kx = KX.astype(np.float32, copy=False).ravel()
        ky = KY.astype(np.float32, copy=False).ravel()
        self.k_max = float(max(np.max(np.abs(kx)), np.max(np.abs(ky))))
        _write_buffer(self.handles.device, self._buffers["kx"][1], kx)
        _write_buffer(self.handles.device, self._buffers["ky"][1], ky)

        kmag = np.sqrt(KX * KX + KY * KY)
        mid = (kmag > self.cfg.k_cut) & (kmag <= self.cfg.resid_mid_cut)
        high = (kmag > self.cfg.resid_mid_cut)
        self._mid_idx = np.flatnonzero(mid).astype(np.uint32)
        self._n_mid = int(self._mid_idx.size)
        self._n_high = int(np.count_nonzero(high))
        if self._n_mid > 0:
            self._alloc_or_update_index_buffer("mid_idx", self._mid_idx)
            self._alloc_or_update_float_buffer("mid_scores", self._n_mid)
            max_partials = (self._n_mid + 255) // 256
            self._alloc_or_update_uvec2_buffer("max_partials", max_partials)
            self._alloc_or_update_uvec2_buffer("max_best", 1)
        else:
            self._buffers["mid_idx"] = None
            self._buffers["mid_scores"] = None
            self._buffers["max_partials"] = None
            self._buffers["max_best"] = None

    def _init_fft_plan(self):
        dummy = np.zeros((self.N, self.N), dtype=np.complex64)
        self.plan = self.fft._get_plan(dummy, direction="fft")  # type: ignore[attr-defined]
        if self.plan is None:
            raise RuntimeError("vkFFT plan unavailable for encode_proxy")

    def _build_pipelines(self):
        shaders = [
            ("real_to_complex", "real_to_complex", 4),
            ("spectral_truncation", "spectral_truncation", 16),
            ("smooth_x", "decode_smooth_x", 12),
            ("smooth_y", "decode_smooth_y", 12),
            ("diff", "encode_diff", 4),
            ("gather_complex", "encode_gather_complex", 4),
            ("gather_mag2", "encode_gather_mag2", 4),
            ("band_energy", "encode_band_energy", 16),
            ("reduce_sum", "reduce_sum", 4),
            ("reduce_max_idx", "reduce_max_idx", 4),
            ("reduce_max_finalize", "reduce_max_finalize", 4),
            ("select_topk", "encode_select_topk", 8),
        ]
        for name, shader_name, push_size in shaders:
            shader_path = resolve_shader(shader_name)
            spv_path = resolve_spv(shader_name)
            compile_shader(shader_path, spv_path)
            pipeline = self._make_pipeline(name, shader_path, spv_path, push_size, self._binding_count_for(name))
            self._pipelines[name] = pipeline
        self._max_descriptor_bindings = max(self._binding_count_for(name) for name, _, _ in shaders)

    def _binding_count_for(self, name: str) -> int:
        return {
            "real_to_complex": 2,
            "spectral_truncation": 3,
            "smooth_x": 2,
            "smooth_y": 2,
            "diff": 3,
            "gather_complex": 3,
            "gather_mag2": 3,
            "band_energy": 5,
            "reduce_sum": 2,
            "reduce_max_idx": 2,
            "reduce_max_finalize": 2,
            "select_topk": 4,
        }[name]

    def _make_pipeline(self, name: str, shader_path: Path, spv_path: Path, push_size: int, bindings: int) -> _Pipeline:
        device = self.handles.device
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

    def _submit_and_wait(self, cmd, query_pool=None):
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vk.vkCreateFence(self.handles.device, fence_info, None)
        vk.vkQueueSubmit(self.handles.queue, 1, [submit_info], fence)
        t0 = time.perf_counter()
        vk.vkWaitForFences(self.handles.device, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        wait_ms = 1000 * (time.perf_counter() - t0)
        if self._timing_active:
            self._timing_last["fence_wait_ms"] += wait_ms
            self._timing_last["gpu_wait_ms"] += wait_ms
        vk.vkDestroyFence(self.handles.device, fence, None)
        if query_pool is not None:
            if self._timing_active:
                self._timing_last["gpu_time_ms"] += self._read_timestamp_ms(query_pool)
            vk.vkDestroyQueryPool(self.handles.device, query_pool, None)

    def _queue_wait_idle(self) -> None:
        t0 = time.perf_counter()
        vk.vkQueueWaitIdle(self.handles.queue)
        wait_ms = 1000 * (time.perf_counter() - t0)
        if self._timing_active:
            self._timing_last["queue_wait_ms"] += wait_ms
            self._timing_last["gpu_wait_ms"] += wait_ms

    def _ensure_descriptor_pool(self) -> object:
        if self._descriptor_pool is not None:
            return self._descriptor_pool
        device = self.handles.device
        max_sets = 64
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=self._max_descriptor_bindings * max_sets,
            )
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
            maxSets=max_sets,
        )
        self._descriptor_pool = vk.vkCreateDescriptorPool(device, pool_info, None)
        return self._descriptor_pool

    def _allocate_descriptor_set(self, pipeline: _Pipeline, buffers: Tuple[Tuple[object, int], ...]):
        device = self.handles.device
        descriptor_pool = self._ensure_descriptor_pool()
        vk.vkResetDescriptorPool(device, descriptor_pool, 0)
        descriptor_set = self._allocate_descriptor_set_from_pool(descriptor_pool, pipeline, buffers)
        return descriptor_pool, descriptor_set

    def _allocate_descriptor_set_from_pool(
        self,
        descriptor_pool,
        pipeline: _Pipeline,
        buffers: Tuple[Tuple[object, int], ...],
    ):
        device = self.handles.device
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
        return descriptor_set

    def _dispatch(self, name: str, buffers: Tuple[Tuple[object, int], ...], push_bytes: bytes, groups: Tuple[int, int, int]):
        for buf, nbytes in buffers:
            if buf is None or nbytes == 0:
                return
        pipeline = self._pipelines[name]
        device = self.handles.device
        cmd = self._alloc_command_buffer()
        query_pool = self._create_timestamp_query_pool()

        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        self._cmd_write_timestamps_begin(cmd, query_pool)
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
            if hasattr(vk, "ffi"):
                push_data = vk.ffi.new("char[]", bytes(push_bytes))
            else:
                push_data = bytearray(push_bytes)
            vk.vkCmdPushConstants(
                cmd,
                pipeline.pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_bytes),
                push_data,
            )
        gx, gy, gz = groups
        vk.vkCmdDispatch(cmd, gx, gy, gz)
        self._cmd_write_timestamps_end(cmd, query_pool)
        vk.vkEndCommandBuffer(cmd)

        self._submit_and_wait(cmd, query_pool=query_pool)
        vk.vkFreeCommandBuffers(device, self.command_pool, 1, [cmd])

    def _record_dispatch(
        self,
        cmd,
        name: str,
        buffers: Tuple[Tuple[object, int], ...],
        push_bytes: bytes,
        groups: Tuple[int, int, int],
        *,
        descriptor_pool=None,
    ):
        for buf, nbytes in buffers:
            if buf is None or nbytes == 0:
                return None
        pipeline = self._pipelines[name]
        if descriptor_pool is None:
            _, descriptor_set = self._allocate_descriptor_set(pipeline, buffers)
        else:
            descriptor_set = self._allocate_descriptor_set_from_pool(descriptor_pool, pipeline, buffers)

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
            if hasattr(vk, "ffi"):
                push_data = vk.ffi.new("char[]", bytes(push_bytes))
            else:
                push_data = bytearray(push_bytes) if isinstance(push_bytes, (bytes, bytearray)) else push_bytes
            vk.vkCmdPushConstants(
                cmd,
                pipeline.pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_bytes),
                push_data,
            )
        gx, gy, gz = groups
        vk.vkCmdDispatch(cmd, gx, gy, gz)
        return descriptor_pool

    def _dispatch_batch(self, entries):
        device = self.handles.device
        cmd = self._alloc_command_buffer()
        query_pool = self._create_timestamp_query_pool()
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        self._cmd_write_timestamps_begin(cmd, query_pool)

        total_sets = max(len(entries), 1)
        total_descriptors = 0
        for _, buffers, _, _ in entries:
            if buffers:
                total_descriptors += len(buffers)
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=max(total_descriptors, 1),
            )
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
            maxSets=total_sets,
        )
        descriptor_pool = vk.vkCreateDescriptorPool(device, pool_info, None)

        for idx, entry in enumerate(entries):
            name, buffers, push_bytes, groups = entry
            self._record_dispatch(cmd, name, buffers, push_bytes, groups, descriptor_pool=descriptor_pool)
            if idx < len(entries) - 1:
                barrier = vk.VkMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                )
                vk.vkCmdPipelineBarrier(
                    cmd,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    [barrier],
                    0,
                    None,
                    0,
                    None,
                )

        self._cmd_write_timestamps_end(cmd, query_pool)
        vk.vkEndCommandBuffer(cmd)
        self._submit_and_wait(cmd, query_pool=query_pool)

        vk.vkDestroyDescriptorPool(device, descriptor_pool, None)
        vk.vkFreeCommandBuffers(device, self.command_pool, 1, [cmd])

    # --------------------- public API ---------------------
    def set_mask_low(self, mask_low: np.ndarray) -> None:
        idx = np.flatnonzero(mask_low.ravel()).astype(np.uint32)
        self._mask_low_idx = idx
        self._lowk_count = int(idx.size)
        self._alloc_or_update_index_buffer("lowk_idx", idx)
        self._alloc_or_update_complex_buffer("lowk_vals", self._lowk_count)

    def set_anchor_idx(self, anchor_idx: np.ndarray) -> None:
        idx = np.asarray(anchor_idx, dtype=np.uint32)
        self._anchor_idx = idx
        self._anchor_count = int(idx.size)
        self._alloc_or_update_index_buffer("anchor_idx", idx)
        self._alloc_or_update_complex_buffer("kept_vals", self._anchor_count)

    def encode_proxy(
        self,
        omega: np.ndarray,
        mask_low: np.ndarray,
        anchor_idx: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._timing_reset()
        if self._mask_low_idx is None:
            self.set_mask_low(mask_low)
        if anchor_idx is not None:
            self.set_anchor_idx(anchor_idx)

        omega32 = np.asarray(omega, dtype=np.float32, order="C")
        if omega32.shape != (self.N, self.N):
            raise ValueError(f"omega shape {omega32.shape} does not match N={self.N}")
        _write_buffer(self.handles.device, self._buffers["omega"][1], omega32)

        gx = (self.N + 15) // 16
        gy = (self.N + 15) // 16
        g1d = ((self.total + 255) // 256, 1, 1)

        # base = smooth2d(omega) via separable box filter
        inv_k = 1.0 / float(self.cfg.dashi_smooth_k)
        if self.batch_dispatch:
            self._dispatch_batch(
                [
                    (
                        "smooth_x",
                        (
                            (self._buffers["omega"][0], self._buffers["omega"][2]),
                            (self._buffers["base_a"][0], self._buffers["base_a"][2]),
                        ),
                        struct.pack("<IIf", self.N, int(self.cfg.dashi_smooth_k), float(inv_k)),
                        (gx, gy, 1),
                    ),
                    (
                        "smooth_y",
                        (
                            (self._buffers["base_a"][0], self._buffers["base_a"][2]),
                            (self._buffers["base_b"][0], self._buffers["base_b"][2]),
                        ),
                        struct.pack("<IIf", self.N, int(self.cfg.dashi_smooth_k), float(inv_k)),
                        (gx, gy, 1),
                    ),
                    (
                        "diff",
                        (
                            (self._buffers["omega"][0], self._buffers["omega"][2]),
                            (self._buffers["base_b"][0], self._buffers["base_b"][2]),
                            (self._buffers["residual"][0], self._buffers["residual"][2]),
                        ),
                        struct.pack("<I", self.N),
                        (gx, gy, 1),
                    ),
                ]
            )
        else:
            self._dispatch(
                "smooth_x",
                buffers=(
                    (self._buffers["omega"][0], self._buffers["omega"][2]),
                    (self._buffers["base_a"][0], self._buffers["base_a"][2]),
                ),
                push_bytes=struct.pack("<IIf", self.N, int(self.cfg.dashi_smooth_k), float(inv_k)),
                groups=(gx, gy, 1),
            )
            self._dispatch(
                "smooth_y",
                buffers=(
                    (self._buffers["base_a"][0], self._buffers["base_a"][2]),
                    (self._buffers["base_b"][0], self._buffers["base_b"][2]),
                ),
                push_bytes=struct.pack("<IIf", self.N, int(self.cfg.dashi_smooth_k), float(inv_k)),
                groups=(gx, gy, 1),
            )

            # residual = omega - base
            self._dispatch(
                "diff",
                buffers=(
                    (self._buffers["omega"][0], self._buffers["omega"][2]),
                    (self._buffers["base_b"][0], self._buffers["base_b"][2]),
                    (self._buffers["residual"][0], self._buffers["residual"][2]),
                ),
                push_bytes=struct.pack("<I", self.N),
                groups=(gx, gy, 1),
            )

        # omega_hat from omega
        self._dispatch(
            "real_to_complex",
            buffers=(
                (self._buffers["omega"][0], self._buffers["omega"][2]),
                (self.plan.device_buffer, self.plan.bytes_len),  # type: ignore[attr-defined]
            ),
            push_bytes=struct.pack("<I", self.total),
            groups=g1d,
        )
        self.fft._run_vkfft(self.plan, inverse=False)  # type: ignore[attr-defined]
        if self._timing_active:
            vkfft_ms = self.fft.get_last_timings().get("vkfft_gpu_time_ms", 0.0)
            self._timing_last["gpu_time_ms"] += float(vkfft_ms)
        self._queue_wait_idle()

        # gather low-k coeffs
        self._dispatch(
            "gather_complex",
            buffers=(
                (self.plan.device_buffer, self.plan.bytes_len),  # type: ignore[attr-defined]
                (self._buffers["lowk_idx"][0], self._buffers["lowk_idx"][2]),
                (self._buffers["lowk_vals"][0], self._buffers["lowk_vals"][2]),
            ),
            push_bytes=struct.pack("<I", self._lowk_count),
            groups=((self._lowk_count + 255) // 256, 1, 1),
        )

        # R_hat from residual
        self._dispatch(
            "real_to_complex",
            buffers=(
                (self._buffers["residual"][0], self._buffers["residual"][2]),
                (self.plan.device_buffer, self.plan.bytes_len),  # type: ignore[attr-defined]
            ),
            push_bytes=struct.pack("<I", self.total),
            groups=g1d,
        )
        self.fft._run_vkfft(self.plan, inverse=False)  # type: ignore[attr-defined]
        if self._timing_active:
            vkfft_ms = self.fft.get_last_timings().get("vkfft_gpu_time_ms", 0.0)
            self._timing_last["gpu_time_ms"] += float(vkfft_ms)
        self._queue_wait_idle()

        if self.spectral_truncation != "none":
            self._dispatch(
                "spectral_truncation",
                buffers=(
                    (self.plan.device_buffer, self.plan.bytes_len),  # type: ignore[attr-defined]
                    (self._buffers["kx"][0], self._buffers["kx"][2]),
                    (self._buffers["ky"][0], self._buffers["ky"][2]),
                ),
                push_bytes=struct.pack(
                    "<fffI",
                    float(self.k_max),
                    float(self.trunc_alpha),
                    float(self.trunc_power),
                    self.total,
                ),
                groups=g1d,
            )

        if self._anchor_idx is None:
            self._compute_topk_from_hat(self.plan.device_buffer, self.plan.bytes_len)

        # gather kept anchors from R_hat
        if self._anchor_count:
            self._dispatch(
                "gather_complex",
                buffers=(
                    (self.plan.device_buffer, self.plan.bytes_len),  # type: ignore[attr-defined]
                    (self._buffers["anchor_idx"][0], self._buffers["anchor_idx"][2]),
                    (self._buffers["kept_vals"][0], self._buffers["kept_vals"][2]),
                ),
                push_bytes=struct.pack("<I", self._anchor_count),
                groups=((self._anchor_count + 255) // 256, 1, 1),
            )

        # band energies
        scale_inv2 = 1.0 / float(self.N * self.N) ** 2
        self._dispatch(
            "band_energy",
            buffers=(
                (self.plan.device_buffer, self.plan.bytes_len),  # type: ignore[attr-defined]
                (self._buffers["kx"][0], self._buffers["kx"][2]),
                (self._buffers["ky"][0], self._buffers["ky"][2]),
                (self._buffers["mid_energy"][0], self._buffers["mid_energy"][2]),
                (self._buffers["high_energy"][0], self._buffers["high_energy"][2]),
            ),
            push_bytes=struct.pack(
                "<fffI",
                float(self.cfg.k_cut),
                float(self.cfg.resid_mid_cut),
                float(scale_inv2),
                self.total,
            ),
            groups=g1d,
        )

        sum_mid = self._reduce_sum("mid_energy")
        sum_high = self._reduce_sum("high_energy")

        # read back compact buffers
        lowk_vals = _read_buffer(self.handles.device, self._buffers["lowk_vals"][1], (self._lowk_count,), np.complex64)
        kept_vals = (
            _read_buffer(self.handles.device, self._buffers["kept_vals"][1], (self._anchor_count,), np.complex64)
            if self._anchor_count
            else np.array([], dtype=np.complex64)
        )

        scale = float(self.N * self.N)
        lowk_r = (lowk_vals.real / scale).astype(np.float64)
        lowk_i = (lowk_vals.imag / scale).astype(np.float64)
        kept_r = (kept_vals.real / scale).astype(np.float64)
        kept_i = (kept_vals.imag / scale).astype(np.float64)

        sum_mid_kept = float(np.sum(np.abs(kept_vals) ** 2) / (scale * scale)) if self._anchor_count else 0.0
        rem_mid = self._n_mid - self._anchor_count
        resid_mid_E = float((sum_mid - sum_mid_kept) / rem_mid) if rem_mid > 0 else 0.0
        resid_high_E = float(sum_high / self._n_high) if self._n_high > 0 else 0.0

        header = np.array([0.0, resid_mid_E, resid_high_E, float(self._anchor_count)], dtype=np.float64)
        z = np.concatenate([lowk_r, lowk_i, header, kept_r, kept_i])
        self._timing_finish()
        return z, (self._anchor_idx.copy() if self._anchor_idx is not None else None)

    def encode_proxy_batch(
        self,
        omega_batch: np.ndarray,
        mask_low: np.ndarray,
        anchor_idx: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        omega_arr = np.asarray(omega_batch)
        if omega_arr.ndim == 2:
            z, anchor_idx = self.encode_proxy(omega_arr, mask_low, anchor_idx=anchor_idx)
            return z[None, ...], anchor_idx
        if omega_arr.ndim != 3:
            raise ValueError(f"omega_batch must be 2D or 3D, got shape {omega_arr.shape}")
        zs = []
        timing_totals: Dict[str, float] = {}
        for i in range(omega_arr.shape[0]):
            z, anchor_idx = self.encode_proxy(omega_arr[i], mask_low, anchor_idx=anchor_idx)
            zs.append(z)
            if self.timing_enabled:
                step_timings = self.get_last_timings()
                for key, val in step_timings.items():
                    timing_totals[key] = timing_totals.get(key, 0.0) + float(val)
        if self.timing_enabled:
            self._timing_last = timing_totals
        return np.stack(zs, axis=0), anchor_idx

    # --------------------- internal utils ---------------------
    def _alloc_or_update_index_buffer(self, name: str, idx: np.ndarray) -> None:
        if idx.size == 0:
            entry = self._buffers.get(name)
            if entry:
                buf, mem, _ = entry
                vk.vkDestroyBuffer(self.handles.device, buf, None)
                vk.vkFreeMemory(self.handles.device, mem, None)
            self._buffers[name] = None
            return
        entry = self._buffers.get(name)
        nbytes = idx.nbytes
        if entry is None or entry is False:
            buf, mem = _create_buffer(self.handles.device, self.handles.mem_props, nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
            self._buffers[name] = (buf, mem, nbytes)
        else:
            buf, mem, size = entry
            if size < nbytes:
                vk.vkDestroyBuffer(self.handles.device, buf, None)
                vk.vkFreeMemory(self.handles.device, mem, None)
                buf, mem = _create_buffer(self.handles.device, self.handles.mem_props, nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
                self._buffers[name] = (buf, mem, nbytes)
        _write_buffer(self.handles.device, self._buffers[name][1], idx)

    def _alloc_or_update_complex_buffer(self, name: str, count: int) -> None:
        if count == 0:
            entry = self._buffers.get(name)
            if entry:
                buf, mem, _ = entry
                vk.vkDestroyBuffer(self.handles.device, buf, None)
                vk.vkFreeMemory(self.handles.device, mem, None)
            self._buffers[name] = None
            return
        nbytes = count * 8
        entry = self._buffers.get(name)
        if entry is None or entry is False:
            buf, mem = _create_buffer(self.handles.device, self.handles.mem_props, nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
            self._buffers[name] = (buf, mem, nbytes)
        else:
            buf, mem, size = entry
            if size < nbytes:
                vk.vkDestroyBuffer(self.handles.device, buf, None)
                vk.vkFreeMemory(self.handles.device, mem, None)
                buf, mem = _create_buffer(self.handles.device, self.handles.mem_props, nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
                self._buffers[name] = (buf, mem, nbytes)

    def _alloc_or_update_float_buffer(self, name: str, count: int) -> None:
        if count == 0:
            entry = self._buffers.get(name)
            if entry:
                buf, mem, _ = entry
                vk.vkDestroyBuffer(self.handles.device, buf, None)
                vk.vkFreeMemory(self.handles.device, mem, None)
            self._buffers[name] = None
            return
        nbytes = count * 4
        entry = self._buffers.get(name)
        if entry is None or entry is False:
            buf, mem = _create_buffer(self.handles.device, self.handles.mem_props, nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
            self._buffers[name] = (buf, mem, nbytes)
        else:
            buf, mem, size = entry
            if size < nbytes:
                vk.vkDestroyBuffer(self.handles.device, buf, None)
                vk.vkFreeMemory(self.handles.device, mem, None)
                buf, mem = _create_buffer(self.handles.device, self.handles.mem_props, nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
                self._buffers[name] = (buf, mem, nbytes)

    def _alloc_or_update_uvec2_buffer(self, name: str, count: int) -> None:
        if count == 0:
            entry = self._buffers.get(name)
            if entry:
                buf, mem, _ = entry
                vk.vkDestroyBuffer(self.handles.device, buf, None)
                vk.vkFreeMemory(self.handles.device, mem, None)
            self._buffers[name] = None
            return
        nbytes = count * 8
        entry = self._buffers.get(name)
        if entry is None or entry is False:
            buf, mem = _create_buffer(self.handles.device, self.handles.mem_props, nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
            self._buffers[name] = (buf, mem, nbytes)
        else:
            buf, mem, size = entry
            if size < nbytes:
                vk.vkDestroyBuffer(self.handles.device, buf, None)
                vk.vkFreeMemory(self.handles.device, mem, None)
                buf, mem = _create_buffer(self.handles.device, self.handles.mem_props, nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
                self._buffers[name] = (buf, mem, nbytes)

    def _reduce_sum(self, buf_name: str) -> float:
        g1 = (self.partial_len, 1, 1)
        self._dispatch(
            "reduce_sum",
            buffers=(
                (self._buffers[buf_name][0], self._buffers[buf_name][2]),
                (self._buffers["partial0"][0], self._buffers["partial0"][2]),
            ),
            push_bytes=struct.pack("<I", self.total),
            groups=g1,
        )
        reduce_len = self.partial_len
        in_name = "partial0"
        out_name = "partial1"
        while reduce_len > 1:
            out_len = (reduce_len + 255) // 256
            self._dispatch(
                "reduce_sum",
                buffers=(
                    (self._buffers[in_name][0], self._buffers[in_name][2]),
                    (self._buffers[out_name][0], self._buffers[out_name][2]),
                ),
                push_bytes=struct.pack("<I", reduce_len),
                groups=(out_len, 1, 1),
            )
            reduce_len = out_len
            in_name, out_name = out_name, in_name
        scalar = _read_buffer(self.handles.device, self._buffers[in_name][1], (1,), np.float32)
        return float(scalar[0])

    def _compute_topk_from_hat(self, hat_buf: object, hat_bytes: int) -> None:
        topk = min(self.cfg.topk_mid, self._n_mid)
        if topk <= 0 or self._n_mid == 0:
            self._anchor_idx = np.array([], dtype=np.uint32)
            self._anchor_count = 0
            self._alloc_or_update_index_buffer("anchor_idx", self._anchor_idx)
            self._alloc_or_update_complex_buffer("kept_vals", 0)
            return

        if self._buffers["anchor_idx"] is None:
            self._alloc_or_update_index_buffer("anchor_idx", np.zeros((topk,), dtype=np.uint32))
        self._anchor_count = int(topk)

        self._dispatch(
            "gather_mag2",
            buffers=(
                (hat_buf, hat_bytes),
                (self._buffers["mid_idx"][0], self._buffers["mid_idx"][2]),
                (self._buffers["mid_scores"][0], self._buffers["mid_scores"][2]),
            ),
            push_bytes=struct.pack("<I", self._n_mid),
            groups=((self._n_mid + 255) // 256, 1, 1),
        )

        gmax = ((self._n_mid + 255) // 256, 1, 1)
        for k in range(self._anchor_count):
            self._dispatch(
                "reduce_max_idx",
                buffers=(
                    (self._buffers["mid_scores"][0], self._buffers["mid_scores"][2]),
                    (self._buffers["max_partials"][0], self._buffers["max_partials"][2]),
                ),
                push_bytes=struct.pack("<I", self._n_mid),
                groups=gmax,
            )
            self._dispatch(
                "reduce_max_finalize",
                buffers=(
                    (self._buffers["max_partials"][0], self._buffers["max_partials"][2]),
                    (self._buffers["max_best"][0], self._buffers["max_best"][2]),
                ),
                push_bytes=struct.pack("<I", gmax[0]),
                groups=(1, 1, 1),
            )
            self._dispatch(
                "select_topk",
                buffers=(
                    (self._buffers["mid_scores"][0], self._buffers["mid_scores"][2]),
                    (self._buffers["mid_idx"][0], self._buffers["mid_idx"][2]),
                    (self._buffers["max_best"][0], self._buffers["max_best"][2]),
                    (self._buffers["anchor_idx"][0], self._buffers["anchor_idx"][2]),
                ),
                push_bytes=struct.pack("<II", k, self._n_mid),
                groups=(1, 1, 1),
            )

        self._anchor_idx = _read_buffer(self.handles.device, self._buffers["anchor_idx"][1], (self._anchor_count,), np.uint32)
        self._alloc_or_update_complex_buffer("kept_vals", self._anchor_count)
