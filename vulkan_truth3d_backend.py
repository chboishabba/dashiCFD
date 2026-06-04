from __future__ import annotations

"""SPV/vkFFT backend for 3D periodic incompressible truth generation."""

import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

CORE_ROOT = Path(__file__).resolve().parent / "dashiCORE"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from gpu_common_methods import compile_shader, resolve_shader, resolve_spv  # type: ignore
from gpu_vkfft_adapter import VkFFTExecutor  # type: ignore
from gpu_vulkan_dispatcher import (  # type: ignore
    HOST_VISIBLE_COHERENT,
    VulkanHandles,
    VulkanDispatchConfig,
    _create_buffer,
    _read_buffer,
    _write_buffer,
    create_vulkan_handles,
)

try:
    import vulkan as vk  # type: ignore
except Exception as exc:  # pragma: no cover - only hit when Vulkan is missing
    vk = None  # type: ignore
    _VK_IMPORT_ERROR = exc
else:
    _VK_IMPORT_ERROR = None


Array = np.ndarray


@dataclass
class _Pipeline:
    name: str
    shader_path: Path
    spv_path: Path
    descriptor_set_layout: object
    pipeline_layout: object
    pipeline: object
    shader_module: object
    push_size: int


class VulkanTruth3DBackend:
    """GPU-assisted pseudo-spectral 3D velocity solver.

    The state is velocity in Fourier space, stored as three complex64 scalar
    component buffers.  Nonlinear RHS construction, Leray projection, curl,
    dealiasing, and RK2 combine are SPV compute kernels; 3D FFT/IFFT work is
    delegated to the vendored vkFFT adapter.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        nu0: float,
        length: float,
        fft_backend: str = "vkfft-vulkan",
        timing_enabled: bool = True,
    ) -> None:
        if vk is None:
            raise RuntimeError(f"vulkan python package not available: {_VK_IMPORT_ERROR}")
        self.N = int(n)
        self.total = int(n * n * n)
        self.dt = float(dt)
        self.nu0 = float(nu0)
        self.length = float(length)
        self.cutoff = (float(n) / 3.0) * (2.0 * np.pi / float(length))
        self.timing_enabled = bool(timing_enabled)
        self._timing_active = False
        self._timing_last: Dict[str, float] = {
            "gpu_time_ms": 0.0,
            "gpu_wait_ms": 0.0,
            "fence_wait_ms": 0.0,
            "queue_wait_ms": 0.0,
        }

        self.handles: VulkanHandles = create_vulkan_handles()
        self.command_pool = self._create_command_pool()
        self.fft_ifft = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend, timing_enabled=timing_enabled)
        self.fft_fwd = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend, timing_enabled=timing_enabled)
        self.fft_backend = fft_backend
        self._shell_lut_active = False

        self._pipelines: Dict[str, _Pipeline] = {}
        self._buffers: Dict[str, Tuple[object, object, int]] = {}
        self._build_pipelines()
        self._alloc_buffers()
        self._init_k_buffers()
        self._init_fft_plans()

    # --------------------- setup ---------------------
    def _create_command_pool(self):
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.handles.queue_family_index,
        )
        return vk.vkCreateCommandPool(self.handles.device, pool_info, None)

    def _alloc_buffers(self) -> None:
        usage = (
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )

        def buf(name: str, nbytes: int) -> None:
            self._buffers[name] = _create_buffer(
                self.handles.device,
                self.handles.mem_props,
                nbytes,
                usage,
                HOST_VISIBLE_COHERENT,
            ) + (nbytes,)

        real_bytes = self.total * 4
        complex_bytes = self.total * 8

        for prefix in ("u", "mid", "next", "rhs1", "rhs2", "adv_hat", "omega_hat"):
            for comp in "xyz":
                buf(f"{prefix}_{comp}", complex_bytes)
        for comp in "xyz":
            buf(f"u_real_{comp}", real_bytes)
            buf(f"adv_{comp}", real_bytes)
        for comp in "xyz":
            for deriv in "xyz":
                buf(f"d{comp}_d{deriv}", real_bytes)
        for name in ("kx", "ky", "kz", "k2"):
            buf(name, real_bytes)
        buf("shell_id", self.total * 4)

    def _init_k_buffers(self) -> None:
        dx = self.length / float(self.N)
        k = np.fft.fftfreq(self.N, d=dx) * 2.0 * np.pi
        kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
        k2 = kx * kx + ky * ky + kz * kz
        for name, arr in (("kx", kx), ("ky", ky), ("kz", kz), ("k2", k2)):
            _write_buffer(self.handles.device, self._buffers[name][1], np.asarray(arr, dtype=np.float32).ravel())
        _write_buffer(self.handles.device, self._buffers["shell_id"][1], np.zeros(self.total, dtype=np.uint32))

    def _init_fft_plans(self) -> None:
        dummy = np.zeros((self.N, self.N, self.N), dtype=np.complex64)
        self.ifft_plan = self.fft_ifft._get_plan(dummy, direction="ifft")  # type: ignore[attr-defined]
        if self.ifft_plan is None:
            raise RuntimeError("vkFFT 3D inverse plan unavailable")
        self.fft_plan = self.fft_fwd._get_plan(dummy, direction="fft")  # type: ignore[attr-defined]
        if self.fft_plan is None:
            raise RuntimeError("vkFFT 3D forward plan unavailable")

    def _build_pipelines(self) -> None:
        shaders = [
            ("copy", "complex_copy_3d", 4, 2),
            ("r2c", "real_to_complex_3d", 4, 2),
            ("c2r", "complex_to_real_3d", 8, 2),
            ("derivative", "derivative_hat_3d", 4, 3),
            ("project", "leray_project_3d", 4, 7),
            ("dealias", "dealias_3d", 8, 6),
            ("shell_filter", "shell_filter_3d", 16, 6),
            ("curl", "curl_3d", 4, 9),
            ("advect", "advect_vector_3d", 4, 15),
            ("rhs", "rhs_projected_ns_3d", 12, 13),
            ("combine", "combine_vector_hat_3d", 16, 16),
        ]
        for name, shader_name, push_size, bindings in shaders:
            shader_path = resolve_shader(shader_name)
            spv_path = resolve_spv(shader_name)
            compile_shader(shader_path, spv_path)
            self._pipelines[name] = self._make_pipeline(name, shader_path, spv_path, push_size, bindings)

    def _make_pipeline(self, name: str, shader_path: Path, spv_path: Path, push_size: int, bindings: int) -> _Pipeline:
        binding_layouts = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            )
            for b in range(bindings)
        ]
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(binding_layouts),
            pBindings=binding_layouts,
        )
        descriptor_set_layout = vk.vkCreateDescriptorSetLayout(self.handles.device, layout_info, None)
        push_range = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=push_size)
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_range],
        )
        pipeline_layout = vk.vkCreatePipelineLayout(self.handles.device, pipeline_layout_info, None)
        code_bytes = spv_path.read_bytes()
        shader_module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code_bytes),
            pCode=code_bytes,
        )
        shader_module = vk.vkCreateShaderModule(self.handles.device, shader_module_info, None)
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
        pipeline = vk.vkCreateComputePipelines(self.handles.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)[0]
        return _Pipeline(name, shader_path, spv_path, descriptor_set_layout, pipeline_layout, pipeline, shader_module, push_size)

    # --------------------- Vulkan helpers ---------------------
    def _buf(self, name: str) -> Tuple[object, int]:
        b, _m, nbytes = self._buffers[name]
        return b, nbytes

    def _alloc_command_buffer(self):
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vk.vkAllocateCommandBuffers(self.handles.device, alloc_info)[0]

    def _submit_and_wait(self, cmd) -> None:
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
        wait_ms = 1000.0 * (time.perf_counter() - t0)
        if self._timing_active:
            self._timing_last["fence_wait_ms"] += wait_ms
            self._timing_last["gpu_wait_ms"] += wait_ms
        vk.vkDestroyFence(self.handles.device, fence, None)

    def _queue_wait_idle(self) -> None:
        t0 = time.perf_counter()
        vk.vkQueueWaitIdle(self.handles.queue)
        wait_ms = 1000.0 * (time.perf_counter() - t0)
        if self._timing_active:
            self._timing_last["queue_wait_ms"] += wait_ms
            self._timing_last["gpu_wait_ms"] += wait_ms

    def _copy_buffer(self, src, dst, nbytes: int) -> None:
        cmd = self._alloc_command_buffer()
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        vk.vkCmdCopyBuffer(cmd, src, dst, 1, [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=nbytes)])
        vk.vkEndCommandBuffer(cmd)
        self._submit_and_wait(cmd)
        vk.vkFreeCommandBuffers(self.handles.device, self.command_pool, 1, [cmd])

    def _allocate_descriptor_set(self, pipeline: _Pipeline, buffers: Tuple[Tuple[object, int], ...]):
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=1,
            pPoolSizes=[
                vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=len(buffers))
            ],
            maxSets=1,
        )
        descriptor_pool = vk.vkCreateDescriptorPool(self.handles.device, pool_info, None)
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[pipeline.descriptor_set_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(self.handles.device, alloc_info)[0]
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
        vk.vkUpdateDescriptorSets(self.handles.device, len(writes), writes, 0, None)
        return descriptor_pool, descriptor_set

    def _dispatch(self, name: str, buffers: Tuple[Tuple[object, int], ...], push_bytes: bytes) -> None:
        pipeline = self._pipelines[name]
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
            push_data = vk.ffi.new("char[]", bytes(push_bytes)) if hasattr(vk, "ffi") else bytearray(push_bytes)
            vk.vkCmdPushConstants(
                cmd,
                pipeline.pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_bytes),
                push_data,
            )
        groups = ((self.total + 255) // 256, 1, 1)
        vk.vkCmdDispatch(cmd, groups[0], groups[1], groups[2])
        vk.vkEndCommandBuffer(cmd)
        self._submit_and_wait(cmd)
        vk.vkDestroyDescriptorPool(self.handles.device, descriptor_pool, None)
        vk.vkFreeCommandBuffers(self.handles.device, self.command_pool, 1, [cmd])

    # --------------------- numerical kernels ---------------------
    def set_initial_u_hat(self, u_hat: Array) -> None:
        arr = np.asarray(u_hat, dtype=np.complex64)
        if arr.shape != (self.N, self.N, self.N, 3):
            raise ValueError(f"u_hat shape {arr.shape} does not match {(self.N, self.N, self.N, 3)}")
        for i, comp in enumerate("xyz"):
            _write_buffer(self.handles.device, self._buffers[f"u_{comp}"][1], np.ascontiguousarray(arr[..., i].ravel()))
        self._project_dealias_prefix("u")

    def set_velocity_real(self, u: Array, *, project: bool = False, dealias: bool = False) -> None:
        """Upload real velocity and transform it to the resident spectral state."""
        arr = np.asarray(u, dtype=np.float32)
        if arr.shape != (self.N, self.N, self.N, 3):
            raise ValueError(f"velocity shape {arr.shape} does not match {(self.N, self.N, self.N, 3)}")
        for i, comp in enumerate("xyz"):
            _write_buffer(self.handles.device, self._buffers[f"u_real_{comp}"][1], np.ascontiguousarray(arr[..., i].ravel()))
            self._run_fft_real_to_hat(f"u_real_{comp}", f"u_{comp}")
        if project:
            self._dispatch(
                "project",
                (
                    self._buf("u_x"),
                    self._buf("u_y"),
                    self._buf("u_z"),
                    self._buf("kx"),
                    self._buf("ky"),
                    self._buf("kz"),
                    self._buf("k2"),
                ),
                struct.pack("<I", self.total),
            )
        if dealias:
            self._dispatch(
                "dealias",
                (self._buf("u_x"), self._buf("u_y"), self._buf("u_z"), self._buf("kx"), self._buf("ky"), self._buf("kz")),
                struct.pack("<If", self.total, float(self.cutoff)),
            )

    def set_omega_real(self, omega: Array) -> None:
        """Upload real vorticity and transform it to resident spectral buffers."""
        arr = np.asarray(omega, dtype=np.float32)
        if arr.shape != (self.N, self.N, self.N, 3):
            raise ValueError(f"omega shape {arr.shape} does not match {(self.N, self.N, self.N, 3)}")
        for i, comp in enumerate("xyz"):
            _write_buffer(self.handles.device, self._buffers[f"u_real_{comp}"][1], np.ascontiguousarray(arr[..., i].ravel()))
            self._run_fft_real_to_hat(f"u_real_{comp}", f"omega_hat_{comp}")

    def set_shell_ids(self, shell_ids: Array) -> None:
        """Upload exact CPU shell labels for parity-safe GPU shell filtering."""
        arr = np.asarray(shell_ids)
        if arr.shape != (self.N, self.N, self.N):
            raise ValueError(f"shell_ids shape {arr.shape} does not match {(self.N, self.N, self.N)}")
        if np.any(arr < 0):
            raise ValueError("shell_ids must be non-negative")
        _write_buffer(
            self.handles.device,
            self._buffers["shell_id"][1],
            np.ascontiguousarray(arr.astype(np.uint32, copy=False).ravel()),
        )
        self._shell_lut_active = True

    def _project_dealias_prefix(self, prefix: str) -> None:
        self._dispatch(
            "project",
            (self._buf(f"{prefix}_x"), self._buf(f"{prefix}_y"), self._buf(f"{prefix}_z"), self._buf("kx"), self._buf("ky"), self._buf("kz"), self._buf("k2")),
            struct.pack("<I", self.total),
        )
        self._dispatch(
            "dealias",
            (self._buf(f"{prefix}_x"), self._buf(f"{prefix}_y"), self._buf(f"{prefix}_z"), self._buf("kx"), self._buf("ky"), self._buf("kz")),
            struct.pack("<If", self.total, float(self.cutoff)),
        )

    def _run_ifft_to_real(self, source_complex: str, out_real: str) -> None:
        self._copy_buffer(self._buf(source_complex)[0], self.ifft_plan.device_buffer, self.ifft_plan.bytes_len)
        self.fft_ifft._run_vkfft(self.ifft_plan, inverse=True)  # type: ignore[attr-defined]
        if self._timing_active:
            self._timing_last["gpu_time_ms"] += float(self.fft_ifft.get_last_timings().get("vkfft_gpu_time_ms", 0.0))
        self._queue_wait_idle()
        self._dispatch(
            "c2r",
            ((self.ifft_plan.device_buffer, self.ifft_plan.bytes_len), self._buf(out_real)),
            struct.pack("<If", self.total, 1.0),
        )

    def _run_derivative_to_real(self, source_complex: str, k_name: str, out_real: str) -> None:
        self._dispatch(
            "derivative",
            (self._buf(source_complex), self._buf(k_name), (self.ifft_plan.device_buffer, self.ifft_plan.bytes_len)),
            struct.pack("<I", self.total),
        )
        self.fft_ifft._run_vkfft(self.ifft_plan, inverse=True)  # type: ignore[attr-defined]
        if self._timing_active:
            self._timing_last["gpu_time_ms"] += float(self.fft_ifft.get_last_timings().get("vkfft_gpu_time_ms", 0.0))
        self._queue_wait_idle()
        self._dispatch(
            "c2r",
            ((self.ifft_plan.device_buffer, self.ifft_plan.bytes_len), self._buf(out_real)),
            struct.pack("<If", self.total, 1.0),
        )

    def _run_fft_real_to_hat(self, source_real: str, out_complex: str) -> None:
        self._dispatch(
            "r2c",
            (self._buf(source_real), (self.fft_plan.device_buffer, self.fft_plan.bytes_len)),
            struct.pack("<I", self.total),
        )
        self.fft_fwd._run_vkfft(self.fft_plan, inverse=False)  # type: ignore[attr-defined]
        if self._timing_active:
            self._timing_last["gpu_time_ms"] += float(self.fft_fwd.get_last_timings().get("vkfft_gpu_time_ms", 0.0))
        self._queue_wait_idle()
        self._copy_buffer(self.fft_plan.device_buffer, self._buf(out_complex)[0], self.fft_plan.bytes_len)

    def _filter_shell_complex(self, source_complex: str, out_complex: str, shell: int, convention: str) -> None:
        convention_id = 2 if self._shell_lut_active else (1 if convention == "integer-radius" else 0)
        k_unit = 2.0 * np.pi / float(self.length)
        self._dispatch(
            "shell_filter",
            (
                self._buf(source_complex),
                self._buf(out_complex),
                self._buf("kx"),
                self._buf("ky"),
                self._buf("kz"),
                self._buf("shell_id"),
            ),
            struct.pack("<IIIf", self.total, int(shell), convention_id, float(1.0 / k_unit)),
        )

    def read_shell_vector(self, source_prefix: str, shell: int, convention: str) -> Array:
        """Return a real vector shell projection from resident spectral buffers."""
        comps = []
        for comp in "xyz":
            self._filter_shell_complex(f"{source_prefix}_{comp}", f"mid_{comp}", int(shell), convention)
            self._run_ifft_to_real(f"mid_{comp}", f"u_real_{comp}")
            comps.append(
                _read_buffer(self.handles.device, self._buffers[f"u_real_{comp}"][1], (self.N, self.N, self.N), np.float32)
            )
        return np.stack(comps, axis=-1)

    def read_shell_scalar_from_real(self, field: Array, shell: int, convention: str) -> Array:
        """Upload a scalar real field, shell-filter it spectrally, and read it back."""
        arr = np.asarray(field, dtype=np.float32)
        if arr.shape != (self.N, self.N, self.N):
            raise ValueError(f"scalar shape {arr.shape} does not match {(self.N, self.N, self.N)}")
        _write_buffer(self.handles.device, self._buffers["adv_x"][1], np.ascontiguousarray(arr.ravel()))
        self._run_fft_real_to_hat("adv_x", "adv_hat_x")
        self._filter_shell_complex("adv_hat_x", "mid_x", int(shell), convention)
        self._run_ifft_to_real("mid_x", "adv_x")
        return _read_buffer(self.handles.device, self._buffers["adv_x"][1], (self.N, self.N, self.N), np.float32)

    def read_velocity_derivatives(
        self,
        source_prefix: str = "u",
        *,
        k_names: Tuple[str, str, str] = ("kx", "ky", "kz"),
    ) -> dict[tuple[int, int], Array]:
        """Return real spectral derivatives d u_component / d coordinate."""
        out: dict[tuple[int, int], Array] = {}
        comps = "xyz"
        for i, comp in enumerate(comps):
            for j, deriv in enumerate(comps):
                name = f"d{comp}_d{deriv}"
                self._run_derivative_to_real(f"{source_prefix}_{comp}", k_names[j], name)
                out[(i, j)] = _read_buffer(self.handles.device, self._buffers[name][1], (self.N, self.N, self.N), np.float32)
        return out

    def read_shell_velocity_derivatives(
        self,
        shell: int,
        convention: str,
        *,
        k_names: Tuple[str, str, str] = ("kx", "ky", "kz"),
    ) -> dict[tuple[int, int], Array]:
        """Return real derivatives of the shell-filtered resident velocity."""
        out: dict[tuple[int, int], Array] = {}
        comps = "xyz"
        for i, comp in enumerate(comps):
            self._filter_shell_complex(f"u_{comp}", f"mid_{comp}", int(shell), convention)
            for j, deriv in enumerate(comps):
                name = f"d{comp}_d{deriv}"
                self._run_derivative_to_real(f"mid_{comp}", k_names[j], name)
                out[(i, j)] = _read_buffer(self.handles.device, self._buffers[name][1], (self.N, self.N, self.N), np.float32)
        return out

    def _compute_rhs(self, source_prefix: str, rhs_prefix: str) -> None:
        for comp in "xyz":
            self._run_ifft_to_real(f"{source_prefix}_{comp}", f"u_real_{comp}")
        for comp in "xyz":
            for deriv in "xyz":
                self._run_derivative_to_real(f"{source_prefix}_{comp}", f"k{deriv}", f"d{comp}_d{deriv}")
        self._dispatch(
            "advect",
            (
                self._buf("u_real_x"), self._buf("u_real_y"), self._buf("u_real_z"),
                self._buf("dx_dx"), self._buf("dx_dy"), self._buf("dx_dz"),
                self._buf("dy_dx"), self._buf("dy_dy"), self._buf("dy_dz"),
                self._buf("dz_dx"), self._buf("dz_dy"), self._buf("dz_dz"),
                self._buf("adv_x"), self._buf("adv_y"), self._buf("adv_z"),
            ),
            struct.pack("<I", self.total),
        )
        for comp in "xyz":
            self._run_fft_real_to_hat(f"adv_{comp}", f"adv_hat_{comp}")
        self._dispatch(
            "rhs",
            (
                self._buf("adv_hat_x"), self._buf("adv_hat_y"), self._buf("adv_hat_z"),
                self._buf(f"{source_prefix}_x"), self._buf(f"{source_prefix}_y"), self._buf(f"{source_prefix}_z"),
                self._buf("kx"), self._buf("ky"), self._buf("kz"), self._buf("k2"),
                self._buf(f"{rhs_prefix}_x"), self._buf(f"{rhs_prefix}_y"), self._buf(f"{rhs_prefix}_z"),
            ),
            struct.pack("<Iff", self.total, float(self.nu0), float(self.cutoff)),
        )

    def _combine(self, source_prefix: str, rhs1_prefix: str, rhs2_prefix: str, out_prefix: str, use_two_rhs: bool) -> None:
        self._dispatch(
            "combine",
            (
                self._buf(f"{source_prefix}_x"), self._buf(f"{source_prefix}_y"), self._buf(f"{source_prefix}_z"),
                self._buf(f"{rhs1_prefix}_x"), self._buf(f"{rhs1_prefix}_y"), self._buf(f"{rhs1_prefix}_z"),
                self._buf(f"{rhs2_prefix}_x"), self._buf(f"{rhs2_prefix}_y"), self._buf(f"{rhs2_prefix}_z"),
                self._buf("kx"), self._buf("ky"), self._buf("kz"), self._buf("k2"),
                self._buf(f"{out_prefix}_x"), self._buf(f"{out_prefix}_y"), self._buf(f"{out_prefix}_z"),
            ),
            struct.pack("<IIff", self.total, 1 if use_two_rhs else 0, float(self.dt), float(self.cutoff)),
        )

    def step(self) -> None:
        self._timing_active = bool(self.timing_enabled)
        self._timing_last = {"gpu_time_ms": 0.0, "gpu_wait_ms": 0.0, "fence_wait_ms": 0.0, "queue_wait_ms": 0.0}
        self._compute_rhs("u", "rhs1")
        self._combine("u", "rhs1", "rhs1", "mid", use_two_rhs=False)
        self._compute_rhs("mid", "rhs2")
        self._combine("u", "rhs1", "rhs2", "next", use_two_rhs=True)
        for comp in "xyz":
            self._buffers[f"u_{comp}"], self._buffers[f"next_{comp}"] = self._buffers[f"next_{comp}"], self._buffers[f"u_{comp}"]
        self._timing_active = False

    def read_u_hat(self) -> Array:
        comps = []
        for comp in "xyz":
            comps.append(_read_buffer(self.handles.device, self._buffers[f"u_{comp}"][1], (self.N, self.N, self.N), np.complex64))
        return np.stack(comps, axis=-1)

    def read_omega_hat(self) -> Array:
        self._dispatch(
            "curl",
            (
                self._buf("u_x"), self._buf("u_y"), self._buf("u_z"),
                self._buf("kx"), self._buf("ky"), self._buf("kz"),
                self._buf("omega_hat_x"), self._buf("omega_hat_y"), self._buf("omega_hat_z"),
            ),
            struct.pack("<I", self.total),
        )
        comps = []
        for comp in "xyz":
            comps.append(_read_buffer(self.handles.device, self._buffers[f"omega_hat_{comp}"][1], (self.N, self.N, self.N), np.complex64))
        return np.stack(comps, axis=-1)

    def get_last_timings(self) -> Dict[str, float]:
        return dict(self._timing_last)

    def device_info(self) -> Dict[str, object]:
        try:
            props = vk.vkGetPhysicalDeviceProperties(self.handles.physical_device)
            return {
                "device_name": str(getattr(props, "deviceName", "")),
                "vendor_id": int(getattr(props, "vendorID", 0)),
                "device_id": int(getattr(props, "deviceID", 0)),
                "api_version": int(getattr(props, "apiVersion", 0)),
            }
        except Exception:
            return {}

    def runtime_info(self) -> Dict[str, object]:
        return {
            "fft_backend_requested": self.fft_backend,
            "ifft_plan_backend": getattr(self.ifft_plan, "backend", None),
            "fft_plan_backend": getattr(self.fft_plan, "backend", None),
            "complex_dtype": "complex64",
            "real_dtype": "float32",
            "shell_lut_active": self._shell_lut_active,
        }

    def close(self) -> None:
        for executor in (self.fft_ifft, self.fft_fwd):
            for ctx in getattr(executor, "_plans", {}).values():
                # Force the pybind11 VkFFT plan destructor while the Vulkan
                # device is still alive; otherwise interpreter teardown can
                # attempt to release descriptor pools after device destruction.
                ctx.app = None
            executor.close()
        for pipeline in self._pipelines.values():
            vk.vkDestroyPipeline(self.handles.device, pipeline.pipeline, None)
            vk.vkDestroyShaderModule(self.handles.device, pipeline.shader_module, None)
            vk.vkDestroyPipelineLayout(self.handles.device, pipeline.pipeline_layout, None)
            vk.vkDestroyDescriptorSetLayout(self.handles.device, pipeline.descriptor_set_layout, None)
        for buf, mem, _nbytes in self._buffers.values():
            vk.vkDestroyBuffer(self.handles.device, buf, None)
            vk.vkFreeMemory(self.handles.device, mem, None)
        vk.vkDestroyCommandPool(self.handles.device, self.command_pool, None)
        self.handles.close()


class VulkanSpectralDiagnostic3DBackend(VulkanTruth3DBackend):
    """GPU spectral helper for harness diagnostics.

    This class intentionally implements only FFT/shell/derivative operations.
    It is used for CPU/GPU diagnostic parity and does not implement NS time
    stepping.
    """

    def __init__(
        self,
        n: int,
        *,
        length: float,
        fft_backend: str = "vkfft-vulkan",
        precision: str = "float64",
        timing_enabled: bool = True,
    ) -> None:
        if vk is None:
            raise RuntimeError(f"vulkan python package not available: {_VK_IMPORT_ERROR}")
        if precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        self.N = int(n)
        self.total = int(n * n * n)
        self.dt = 0.0
        self.nu0 = 0.0
        self.length = float(length)
        self.cutoff = (float(n) / 3.0) * (2.0 * np.pi / float(length))
        self.timing_enabled = bool(timing_enabled)
        self._timing_active = False
        self._timing_last: Dict[str, float] = {
            "gpu_time_ms": 0.0,
            "gpu_wait_ms": 0.0,
            "fence_wait_ms": 0.0,
            "queue_wait_ms": 0.0,
        }
        self.precision = precision
        self.real_dtype = np.float64 if precision == "float64" else np.float32
        self.complex_dtype = np.complex128 if precision == "float64" else np.complex64
        self._shader_suffix = "_f64" if precision == "float64" else ""
        self._shell_lut_active = False

        self.handles = create_vulkan_handles(
            VulkanDispatchConfig(enable_shader_float64=(precision == "float64"))
        )
        self.command_pool = self._create_command_pool()
        self.fft_ifft = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend, timing_enabled=timing_enabled)
        self.fft_fwd = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend, timing_enabled=timing_enabled)
        self.fft_backend = fft_backend

        self._pipelines: Dict[str, _Pipeline] = {}
        self._buffers: Dict[str, Tuple[object, object, int]] = {}
        self._build_pipelines()
        self._alloc_buffers()
        self._init_k_buffers()
        self._init_fft_plans()

    def _alloc_buffers(self) -> None:
        usage = (
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )

        def buf(name: str, nbytes: int) -> None:
            self._buffers[name] = _create_buffer(
                self.handles.device,
                self.handles.mem_props,
                nbytes,
                usage,
                HOST_VISIBLE_COHERENT,
            ) + (nbytes,)

        real_bytes = self.total * np.dtype(self.real_dtype).itemsize
        complex_bytes = self.total * np.dtype(self.complex_dtype).itemsize
        for prefix in ("u", "mid", "adv_hat", "omega_hat"):
            for comp in "xyz":
                buf(f"{prefix}_{comp}", complex_bytes)
        for comp in "xyz":
            buf(f"u_real_{comp}", real_bytes)
            buf(f"adv_{comp}", real_bytes)
        for comp in "xyz":
            for deriv in "xyz":
                buf(f"d{comp}_d{deriv}", real_bytes)
        for name in ("kx", "ky", "kz", "k2"):
            buf(name, real_bytes)
        buf("shell_id", self.total * 4)

    def _init_k_buffers(self) -> None:
        dx = self.length / float(self.N)
        k = np.fft.fftfreq(self.N, d=dx) * 2.0 * np.pi
        kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
        k2 = kx * kx + ky * ky + kz * kz
        for name, arr in (("kx", kx), ("ky", ky), ("kz", kz), ("k2", k2)):
            _write_buffer(self.handles.device, self._buffers[name][1], np.asarray(arr, dtype=self.real_dtype).ravel())
        _write_buffer(self.handles.device, self._buffers["shell_id"][1], np.zeros(self.total, dtype=np.uint32))

    def _init_fft_plans(self) -> None:
        dummy = np.zeros((self.N, self.N, self.N), dtype=self.complex_dtype)
        self.ifft_plan = self.fft_ifft._get_plan(dummy, direction="ifft")  # type: ignore[attr-defined]
        if self.ifft_plan is None:
            raise RuntimeError(f"vkFFT 3D inverse plan unavailable for {self.complex_dtype}")
        self.fft_plan = self.fft_fwd._get_plan(dummy, direction="fft")  # type: ignore[attr-defined]
        if self.fft_plan is None:
            raise RuntimeError(f"vkFFT 3D forward plan unavailable for {self.complex_dtype}")

    def _build_pipelines(self) -> None:
        suffix = self._shader_suffix
        shaders = [
            ("r2c", f"real_to_complex_3d{suffix}", 4, 2),
            ("c2r", f"complex_to_real_3d{suffix}", 8, 2),
            ("derivative", f"derivative_hat_3d{suffix}", 4, 3),
            ("shell_filter", f"shell_filter_3d{suffix}", 16, 6),
        ]
        for name, shader_name, push_size, bindings in shaders:
            shader_path = resolve_shader(shader_name)
            spv_path = resolve_spv(shader_name)
            compile_shader(shader_path, spv_path)
            self._pipelines[name] = self._make_pipeline(name, shader_path, spv_path, push_size, bindings)

    def set_velocity_real(self, u: Array, *, project: bool = False, dealias: bool = False) -> None:
        if project or dealias:
            raise NotImplementedError("diagnostic spectral backend does not project/dealias uploads")
        arr = np.asarray(u, dtype=self.real_dtype)
        if arr.shape != (self.N, self.N, self.N, 3):
            raise ValueError(f"velocity shape {arr.shape} does not match {(self.N, self.N, self.N, 3)}")
        for i, comp in enumerate("xyz"):
            _write_buffer(self.handles.device, self._buffers[f"u_real_{comp}"][1], np.ascontiguousarray(arr[..., i].ravel()))
            self._run_fft_real_to_hat(f"u_real_{comp}", f"u_{comp}")

    def set_omega_real(self, omega: Array) -> None:
        arr = np.asarray(omega, dtype=self.real_dtype)
        if arr.shape != (self.N, self.N, self.N, 3):
            raise ValueError(f"omega shape {arr.shape} does not match {(self.N, self.N, self.N, 3)}")
        for i, comp in enumerate("xyz"):
            _write_buffer(self.handles.device, self._buffers[f"u_real_{comp}"][1], np.ascontiguousarray(arr[..., i].ravel()))
            self._run_fft_real_to_hat(f"u_real_{comp}", f"omega_hat_{comp}")

    def read_shell_vector(self, source_prefix: str, shell: int, convention: str) -> Array:
        comps = []
        for comp in "xyz":
            self._filter_shell_complex(f"{source_prefix}_{comp}", f"mid_{comp}", int(shell), convention)
            self._run_ifft_to_real(f"mid_{comp}", f"u_real_{comp}")
            comps.append(
                _read_buffer(self.handles.device, self._buffers[f"u_real_{comp}"][1], (self.N, self.N, self.N), self.real_dtype)
            )
        return np.stack(comps, axis=-1)

    def read_shell_scalar_from_real(self, field: Array, shell: int, convention: str) -> Array:
        arr = np.asarray(field, dtype=self.real_dtype)
        if arr.shape != (self.N, self.N, self.N):
            raise ValueError(f"scalar shape {arr.shape} does not match {(self.N, self.N, self.N)}")
        _write_buffer(self.handles.device, self._buffers["adv_x"][1], np.ascontiguousarray(arr.ravel()))
        self._run_fft_real_to_hat("adv_x", "adv_hat_x")
        self._filter_shell_complex("adv_hat_x", "mid_x", int(shell), convention)
        self._run_ifft_to_real("mid_x", "adv_x")
        return _read_buffer(self.handles.device, self._buffers["adv_x"][1], (self.N, self.N, self.N), self.real_dtype)

    def read_velocity_derivatives(
        self,
        source_prefix: str = "u",
        *,
        k_names: Tuple[str, str, str] = ("kx", "ky", "kz"),
    ) -> dict[tuple[int, int], Array]:
        out: dict[tuple[int, int], Array] = {}
        for i, comp in enumerate("xyz"):
            for j, deriv in enumerate("xyz"):
                name = f"d{comp}_d{deriv}"
                self._run_derivative_to_real(f"{source_prefix}_{comp}", k_names[j], name)
                out[(i, j)] = _read_buffer(self.handles.device, self._buffers[name][1], (self.N, self.N, self.N), self.real_dtype)
        return out

    def read_shell_velocity_derivatives(
        self,
        shell: int,
        convention: str,
        *,
        k_names: Tuple[str, str, str] = ("kx", "ky", "kz"),
    ) -> dict[tuple[int, int], Array]:
        out: dict[tuple[int, int], Array] = {}
        for i, comp in enumerate("xyz"):
            self._filter_shell_complex(f"u_{comp}", f"mid_{comp}", int(shell), convention)
            for j, deriv in enumerate("xyz"):
                name = f"d{comp}_d{deriv}"
                self._run_derivative_to_real(f"mid_{comp}", k_names[j], name)
                out[(i, j)] = _read_buffer(self.handles.device, self._buffers[name][1], (self.N, self.N, self.N), self.real_dtype)
        return out

    def runtime_info(self) -> Dict[str, object]:
        return {
            "fft_backend_requested": self.fft_backend,
            "ifft_plan_backend": getattr(self.ifft_plan, "backend", None),
            "fft_plan_backend": getattr(self.fft_plan, "backend", None),
            "complex_dtype": str(np.dtype(self.complex_dtype)),
            "real_dtype": str(np.dtype(self.real_dtype)),
            "shell_lut_active": self._shell_lut_active,
            "diagnostic_only": True,
        }

    def step(self) -> None:
        raise NotImplementedError("VulkanSpectralDiagnostic3DBackend does not implement time stepping")
