#include <xfdtd_cuda/common.cuh>

#include "updator/basic_updator_te.cuh"
#include "updator/basic_updator_te_agency.cuh"
#include "updator/update_scheme.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_DEVICE auto BasicUpdatorTE::updateH() -> void {
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

  // static bool is_print = false;
  // if (!is_print) {
  //   auto block_id = blockIdx.x + blockIdx.y * gridDim.x +
  //                   blockIdx.z * gridDim.x * gridDim.y;
  //   auto thread_id = threadIdx.x + threadIdx.y * blockDim.x +
  //                    threadIdx.z * blockDim.x * blockDim.y;
  //   std::printf(
  //       "blockIdx: (%d, %d, %d), threadIdx: (%d, %d, %d), block_id: %d, "
  //       "thread_id: %d, task:[%lu, %lu), [%lu, %lu), [%lu, %lu)\n",
  //       blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
  //       threadIdx.z, block_id, thread_id, is, ie, js, je, ks, ke);

  //   is_print = true;
  // }

  update<xfdtd::EMF::Attribute::H, Axis::XYZ::X>(
      *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
  update<xfdtd::EMF::Attribute::H, Axis::XYZ::Y>(
      *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
  update<xfdtd::EMF::Attribute::H, Axis::XYZ::Z>(
      *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
}

XFDTD_CUDA_DEVICE auto BasicUpdatorTE::updateE() -> void {
  const auto task = this->task();
  const auto x_range = task.xRange();
  const auto y_range = task.yRange();
  const auto z_range = task.zRange();

  const auto is = x_range.start() == 0 ? 1 : x_range.start();
  const auto ie = x_range.end();
  const auto js = y_range.start() == 0 ? 1 : y_range.start();
  const auto je = y_range.end();
  const auto ks = z_range.start();
  const auto ke = z_range.end();

  update<xfdtd::EMF::Attribute::E, Axis::XYZ::Z>(
      *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke);
}

XFDTD_CUDA_DEVICE auto BasicUpdatorTE::task() const -> IndexTask {
  const auto& node_task = _node_task;
  // blcok
  auto size_x = static_cast<Index>(gridDim.x);
  auto size_y = static_cast<Index>(gridDim.y);
  auto size_z = static_cast<Index>(gridDim.z);
  auto id =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  auto block_task = decomposeTask(node_task, id, size_x, size_y, size_z);
  // thread
  size_x = static_cast<Index>(blockDim.x);
  size_y = static_cast<Index>(blockDim.y);
  size_z = static_cast<Index>(blockDim.z);
  id = threadIdx.x + threadIdx.y * blockDim.x +
       threadIdx.z * blockDim.x * blockDim.y;

  auto thread_task = decomposeTask(block_task, id, size_x, size_y, size_z);
  return thread_task;
}

XFDTD_CUDA_GLOBAL auto __basicUpdatorTEAgencyUpdateH(
    xfdtd::cuda::BasicUpdatorTE* updator) -> void {
  updator->updateH();
}

XFDTD_CUDA_GLOBAL auto __basicUpdatorTEAgencyUpdateE(
    xfdtd::cuda::BasicUpdatorTE* updator) -> void {
  updator->updateE();
}

auto BasicUpdatorTEAgency::updateH(dim3 grid_size, dim3 block_size) -> void {
  __basicUpdatorTEAgencyUpdateH<<<grid_size, block_size>>>(_updator);
}

auto BasicUpdatorTEAgency::updateE(dim3 grid_size, dim3 block_size) -> void {
  __basicUpdatorTEAgencyUpdateE<<<grid_size, block_size>>>(_updator);
}

auto BasicUpdatorTEAgency::setDevice(BasicUpdatorTE* updator) -> void {
  _updator = updator;
}

}  // namespace xfdtd::cuda
