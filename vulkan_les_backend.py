from __future__ import annotations

"""GPU-only 2D vorticity LES stepper using Vulkan + vkFFT."""

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import sys

from dashi_cfd_operator_v4 import make_grid, smooth2d

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


class VulkanLESBackend:
    """GPU-only LES stepper with vkFFT + SPIR-V kernels."""

    def __init__(self, N: int, *, dt: float, nu0: float, Cs: float, fft_backend: str = "vkfft-vulkan"):
        if vk is None:
            raise RuntimeError(f"vulkan python package not available: {_VK_IMPORT_ERROR}")
        self.N = int(N)
        self.total = int(N * N)
        self.dt = float(dt)
        self.nu0 = float(nu0)
        self.Cs = float(Cs)

        self.handles: VulkanHandles = create_vulkan_handles()
        self.command_pool = self._create_command_pool()
        self.fft_omega = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend)
        self.fft_ux = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend)
        self.fft_uy = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend)
        self.fft_lap = VkFFTExecutor(handles=self.handles, fft_backend=fft_backend)

        self._pipelines: Dict[str, _Pipeline] = {}
        self._build_pipelines()
        self._buffers = {}
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
        _buf("omega_tmp", real_bytes)
        _buf("rhs1", real_bytes)
        _buf("rhs2", real_bytes)
        _buf("adv", real_bytes)
        _buf("lap", real_bytes)
        _buf("dwdx", real_bytes)
        _buf("dwdy", real_bytes)
        _buf("ux", real_bytes)
        _buf("uy", real_bytes)
        _buf("nu_t", real_bytes)

        # k-space arrays (float)
        _buf("kx", real_bytes)
        _buf("ky", real_bytes)
        _buf("k2", real_bytes)

        # reduction buffers
        self.partial_len = (self.total + 255) // 256
        _buf("partial0", self.partial_len * 4)
        _buf("partial1", self.partial_len * 4)
        _buf("scalar", 4)

        # extra complex buffers for spectral ops
        complex_bytes = self.total * 8
        _buf("psi_hat", complex_bytes)
        _buf("lap_hat", complex_bytes)

    def _init_k_buffers(self):
        dx, KX, KY, K2 = make_grid(self.N)
        self.dx = float(dx)
        kx = KX.astype(np.float32, copy=False).ravel()
        ky = KY.astype(np.float32, copy=False).ravel()
        k2 = K2.astype(np.float32, copy=False).ravel()
        if k2.size:
            k2[0] = 0.0
        _write_buffer(self.handles.device, self._buffers["kx"][1], kx)
        _write_buffer(self.handles.device, self._buffers["ky"][1], ky)
        _write_buffer(self.handles.device, self._buffers["k2"][1], k2)

    def _init_fft_plans(self):
        dummy = np.zeros((self.N, self.N), dtype=np.complex64)
        self.omega_plan = self.fft_omega._get_plan(dummy, direction="fft")  # type: ignore[attr-defined]
        if self.omega_plan is None:
            raise RuntimeError("vkFFT plan unavailable for omega")
        self.ux_plan = self.fft_ux._get_plan(dummy, direction="ifft")  # type: ignore[attr-defined]
        if self.ux_plan is None:
            raise RuntimeError("vkFFT plan unavailable for ux")
        self.uy_plan = self.fft_uy._get_plan(dummy, direction="ifft")  # type: ignore[attr-defined]
        if self.uy_plan is None:
            raise RuntimeError("vkFFT plan unavailable for uy")
        self.lap_plan = self.fft_lap._get_plan(dummy, direction="ifft")  # type: ignore[attr-defined]
        if self.lap_plan is None:
            raise RuntimeError("vkFFT plan unavailable for lap")

    def _build_pipelines(self):
        shaders = [
            ("real_to_complex", "real_to_complex", 4),
            ("c2r", "decode_complex_to_real", 8),
            ("poisson", "spectral_poisson", 4),
            ("spectral_vel", "spectral_vel", 4),
            ("spectral_laplacian", "spectral_laplacian", 4),
            ("grad_omega", "grad_omega_fd", 8),
            ("advect", "advect", 4),
            ("smagorinsky", "smagorinsky_nu", 16),
            ("rhs", "rhs_comp", 8),
            ("omega_update", "omega_update", 8),
            ("rk2", "rk2_combine", 8),
            ("reduce_sum_sq", "reduce_sum_sq", 4),
            ("reduce_sum", "reduce_sum", 4),
        ]
        for name, shader_name, push_size in shaders:
            shader_path = resolve_shader(shader_name)
            spv_path = resolve_spv(shader_name)
            compile_shader(shader_path, spv_path)
            pipeline = self._make_pipeline(name, shader_path, spv_path, push_size, self._binding_count_for(name))
            self._pipelines[name] = pipeline

    def _binding_count_for(self, name: str) -> int:
        return {
            "real_to_complex": 2,
            "c2r": 2,
            "poisson": 3,
            "spectral_vel": 5,
            "spectral_laplacian": 3,
            "grad_omega": 3,
            "advect": 5,
            "smagorinsky": 3,
            "rhs": 4,
            "omega_update": 3,
            "rk2": 4,
            "reduce_sum_sq": 2,
            "reduce_sum": 2,
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
        vk.vkEndCommandBuffer(cmd)

        self._submit_and_wait(cmd)
        vk.vkDestroyDescriptorPool(device, descriptor_pool, None)
        vk.vkFreeCommandBuffers(device, self.command_pool, 1, [cmd])

    # --------------------- public API ---------------------
    def set_initial_omega(self, omega: np.ndarray) -> None:
        omega32 = np.asarray(omega, dtype=np.float32, order="C")
        if omega32.shape != (self.N, self.N):
            raise ValueError(f"omega shape {omega32.shape} does not match N={self.N}")
        _write_buffer(self.handles.device, self._buffers["omega"][1], omega32)

    def _compute_rhs(self, omega_buf_name: str, rhs_buf_name: str) -> None:
        N = self.N
        total = self.total
        gx = (N + 15) // 16
        gy = (N + 15) // 16
        g1d = ((total + 255) // 256, 1, 1)

        # real -> complex (omega_hat)
        self._dispatch(
            "real_to_complex",
            buffers=(
                (self._buffers[omega_buf_name][0], self._buffers[omega_buf_name][2]),
                (self.omega_plan.device_buffer, self.omega_plan.bytes_len),  # type: ignore[attr-defined]
            ),
            push_bytes=struct.pack("<I", total),
            groups=g1d,
        )

        # FFT omega_hat
        self.fft_omega._run_vkfft(self.omega_plan, inverse=False)  # type: ignore[attr-defined]
        vk.vkQueueWaitIdle(self.handles.queue)

        # psi_hat = -omega_hat / k2
        self._dispatch(
            "poisson",
            buffers=(
                (self.omega_plan.device_buffer, self.omega_plan.bytes_len),  # type: ignore[attr-defined]
                (self._buffers["psi_hat"][0], self._buffers["psi_hat"][2]),
                (self._buffers["k2"][0], self._buffers["k2"][2]),
            ),
            push_bytes=struct.pack("<I", total),
            groups=g1d,
        )

        # ux_hat, uy_hat
        self._dispatch(
            "spectral_vel",
            buffers=(
                (self._buffers["psi_hat"][0], self._buffers["psi_hat"][2]),
                (self.ux_plan.device_buffer, self.ux_plan.bytes_len),  # type: ignore[attr-defined]
                (self.uy_plan.device_buffer, self.uy_plan.bytes_len),  # type: ignore[attr-defined]
                (self._buffers["kx"][0], self._buffers["kx"][2]),
                (self._buffers["ky"][0], self._buffers["ky"][2]),
            ),
            push_bytes=struct.pack("<I", total),
            groups=g1d,
        )

        # iFFT ux_hat, uy_hat
        self.fft_ux._run_vkfft(self.ux_plan, inverse=True)  # type: ignore[attr-defined]
        self.fft_uy._run_vkfft(self.uy_plan, inverse=True)  # type: ignore[attr-defined]
        vk.vkQueueWaitIdle(self.handles.queue)

        # complex -> real (ux, uy)
        scale = np.float32(1.0 / float(N * N))
        self._dispatch(
            "c2r",
            buffers=(
                (self.ux_plan.device_buffer, self.ux_plan.bytes_len),  # type: ignore[attr-defined]
                (self._buffers["ux"][0], self._buffers["ux"][2]),
            ),
            push_bytes=struct.pack("<If", N, float(scale)),
            groups=(gx, gy, 1),
        )
        self._dispatch(
            "c2r",
            buffers=(
                (self.uy_plan.device_buffer, self.uy_plan.bytes_len),  # type: ignore[attr-defined]
                (self._buffers["uy"][0], self._buffers["uy"][2]),
            ),
            push_bytes=struct.pack("<If", N, float(scale)),
            groups=(gx, gy, 1),
        )

        # lap_hat = -k2 * omega_hat
        self._dispatch(
            "spectral_laplacian",
            buffers=(
                (self.omega_plan.device_buffer, self.omega_plan.bytes_len),  # type: ignore[attr-defined]
                (self.lap_plan.device_buffer, self.lap_plan.bytes_len),  # type: ignore[attr-defined]
                (self._buffers["k2"][0], self._buffers["k2"][2]),
            ),
            push_bytes=struct.pack("<I", total),
            groups=g1d,
        )

        # iFFT lap_hat -> lap
        self.fft_lap._run_vkfft(self.lap_plan, inverse=True)  # type: ignore[attr-defined]
        vk.vkQueueWaitIdle(self.handles.queue)
        self._dispatch(
            "c2r",
            buffers=(
                (self.lap_plan.device_buffer, self.lap_plan.bytes_len),  # type: ignore[attr-defined]
                (self._buffers["lap"][0], self._buffers["lap"][2]),
            ),
            push_bytes=struct.pack("<If", N, float(scale)),
            groups=(gx, gy, 1),
        )

        inv_2dx = np.float32(1.0 / (2.0 * self.dx))
        # gradients of omega
        self._dispatch(
            "grad_omega",
            buffers=(
                (self._buffers[omega_buf_name][0], self._buffers[omega_buf_name][2]),
                (self._buffers["dwdx"][0], self._buffers["dwdx"][2]),
                (self._buffers["dwdy"][0], self._buffers["dwdy"][2]),
            ),
            push_bytes=struct.pack("<If", N, float(inv_2dx)),
            groups=(gx, gy, 1),
        )

        # advect term
        self._dispatch(
            "advect",
            buffers=(
                (self._buffers["ux"][0], self._buffers["ux"][2]),
                (self._buffers["uy"][0], self._buffers["uy"][2]),
                (self._buffers["dwdx"][0], self._buffers["dwdx"][2]),
                (self._buffers["dwdy"][0], self._buffers["dwdy"][2]),
                (self._buffers["adv"][0], self._buffers["adv"][2]),
            ),
            push_bytes=struct.pack("<I", N),
            groups=(gx, gy, 1),
        )

        # smagorinsky viscosity
        self._dispatch(
            "smagorinsky",
            buffers=(
                (self._buffers["ux"][0], self._buffers["ux"][2]),
                (self._buffers["uy"][0], self._buffers["uy"][2]),
                (self._buffers["nu_t"][0], self._buffers["nu_t"][2]),
            ),
            push_bytes=struct.pack("<Ifff", N, float(inv_2dx), float(self.Cs), float(self.dx)),
            groups=(gx, gy, 1),
        )

        # rhs = -adv + (nu0+nu_t)*lap
        self._dispatch(
            "rhs",
            buffers=(
                (self._buffers["adv"][0], self._buffers["adv"][2]),
                (self._buffers["lap"][0], self._buffers["lap"][2]),
                (self._buffers["nu_t"][0], self._buffers["nu_t"][2]),
                (self._buffers[rhs_buf_name][0], self._buffers[rhs_buf_name][2]),
            ),
            push_bytes=struct.pack("<If", N, float(self.nu0)),
            groups=(gx, gy, 1),
        )

    def step(self) -> None:
        N = self.N
        gx = (N + 15) // 16
        gy = (N + 15) // 16

        # rhs1 from omega
        self._compute_rhs("omega", "rhs1")

        # omega_tmp = omega + dt * rhs1
        self._dispatch(
            "omega_update",
            buffers=(
                (self._buffers["omega"][0], self._buffers["omega"][2]),
                (self._buffers["rhs1"][0], self._buffers["rhs1"][2]),
                (self._buffers["omega_tmp"][0], self._buffers["omega_tmp"][2]),
            ),
            push_bytes=struct.pack("<If", N, float(self.dt)),
            groups=(gx, gy, 1),
        )

        # rhs2 from omega_tmp
        self._compute_rhs("omega_tmp", "rhs2")

        # omega_next = omega + 0.5*dt*(rhs1+rhs2)
        self._dispatch(
            "rk2",
            buffers=(
                (self._buffers["omega"][0], self._buffers["omega"][2]),
                (self._buffers["rhs1"][0], self._buffers["rhs1"][2]),
                (self._buffers["rhs2"][0], self._buffers["rhs2"][2]),
                (self._buffers["omega_tmp"][0], self._buffers["omega_tmp"][2]),
            ),
            push_bytes=struct.pack("<If", N, float(self.dt)),
            groups=(gx, gy, 1),
        )

        # swap omega <- omega_tmp
        self._buffers["omega"], self._buffers["omega_tmp"] = self._buffers["omega_tmp"], self._buffers["omega"]

    def read_omega(self) -> np.ndarray:
        return _read_buffer(self.handles.device, self._buffers["omega"][1], (self.N, self.N), np.float32)

    def enstrophy(self) -> float:
        total = self.total
        g1 = (self.partial_len, 1, 1)
        self._dispatch(
            "reduce_sum_sq",
            buffers=(
                (self._buffers["omega"][0], self._buffers["omega"][2]),
                (self._buffers["partial0"][0], self._buffers["partial0"][2]),
            ),
            push_bytes=struct.pack("<I", total),
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
        return 0.5 * float(scalar[0]) / float(self.total)


def init_random_omega(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega = smooth2d(rng.standard_normal((N, N)).astype(np.float32), 11)
    omega = (omega - omega.mean()) / (omega.std() + 1e-12)
    return omega.astype(np.float32)
