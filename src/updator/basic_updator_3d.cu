#include <xfdtd_cuda/calculation_param/calculation_param.cuh>

#include "updator/basic_updator_3d.cuh"
#include "updator/basic_updator_3d_agency.cuh"
#include "updator/update_scheme.cuh"
#include "xfdtd_cuda/common.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_DEVICE auto BasicUpdator3D::updateH() -> void {
  const auto task = this->task();
  const auto x_range = task.xRange();
  const auto y_range = task.yRange();
  const auto z_range = task.zRange();

  const auto is = x_range.start();
  const auto ie = x_range.end();
  const auto js = y_range.start();
  const auto je = y_range.end();
  const auto ks = z_range.start();
  const auto ke = z_range.end();

  update<xfdtd::EMF::Attribute::H, Axis::XYZ::X>(
      *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
  update<xfdtd::EMF::Attribute::H, Axis::XYZ::Y>(
      *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
  update<xfdtd::EMF::Attribute::H, Axis::XYZ::Z>(
      *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
}

XFDTD_CUDA_DEVICE auto BasicUpdator3D::updateE() -> void {
  const auto task = this->task();
  const auto x_range = task.xRange();
  const auto y_range = task.yRange();
  const auto z_range = task.zRange();

  {
    // EX
    const auto is = x_range.start();
    const auto ie = x_range.end();
    const auto js = y_range.start() == 0 ? 1 : y_range.start();
    const auto je = y_range.end();
    const auto ks = z_range.start() == 0 ? 1 : z_range.start();
    const auto ke = z_range.end();

    update<xfdtd::EMF::Attribute::E, Axis::XYZ::X>(
        *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
  }

  {
    const auto is = x_range.start() == 0 ? 1 : x_range.start();
    const auto ie = x_range.end();
    const auto js = y_range.start();
    const auto je = y_range.end();
    const auto ks = z_range.start() == 0 ? 1 : z_range.start();
    const auto ke = z_range.end();

    update<xfdtd::EMF::Attribute::E, Axis::XYZ::Y>(
        *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
  }

  // EZ
  {
    const auto is = x_range.start() == 0 ? 1 : x_range.start();
    const auto ie = x_range.end();
    const auto js = y_range.start() == 0 ? 1 : y_range.start();
    const auto je = y_range.end();
    const auto ks = z_range.start();
    const auto ke = z_range.end();

    update<xfdtd::EMF::Attribute::E, Axis::XYZ::Z>(
        *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
  }
}

XFDTD_CUDA_DEVICE auto BasicUpdator3D::task() const -> IndexTask {
  const auto& node_task = _node_task;
  // blcok
  auto size_x = static_cast<Index>(gridDim.x);
  auto size_y = static_cast<Index>(gridDim.y);
  auto size_z = static_cast<Index>(gridDim.z);
  auto id =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  auto block_task = decomposeTask(node_task, id, size_x, size_y, size_z);
  // auto block_task = decomposeTask(node_task, blockIdx.x, blockIdx.y,
  // blockIdx.z, size_x, size_y, size_z);
  // thread
  size_x = static_cast<Index>(blockDim.x);
  size_y = static_cast<Index>(blockDim.y);
  size_z = static_cast<Index>(blockDim.z);
  id = threadIdx.x + threadIdx.y * blockDim.x +
       threadIdx.z * blockDim.x * blockDim.y;

  auto thread_task = decomposeTask(block_task, id, size_x, size_y, size_z);
  // auto thread_task = decomposeTask(block_task, threadIdx.x, threadIdx.y,
  //  threadIdx.z, size_x, size_y, size_z);
  return thread_task;
}

// Agency

XFDTD_CUDA_GLOBAL auto __basicUpdator3DAgencyUpdateH(BasicUpdator3D* updator)
    -> void {
  updator->updateH();
}

XFDTD_CUDA_GLOBAL auto __basicUpdator3DAgencyUpdateE(BasicUpdator3D* updator)
    -> void {
  updator->updateE();
}

auto BasicUpdator3DAgency::updateH(dim3 grid_size, dim3 block_size) -> void {
  __basicUpdator3DAgencyUpdateH<<<grid_size, block_size>>>(_updator);
}

auto BasicUpdator3DAgency::updateE(dim3 grid_size, dim3 block_size) -> void {
  __basicUpdator3DAgencyUpdateE<<<grid_size, block_size>>>(_updator);
}

auto BasicUpdator3DAgency::setDevice(BasicUpdator3D* updator) -> void {
  _updator = updator;
}

}  // namespace xfdtd::cuda
