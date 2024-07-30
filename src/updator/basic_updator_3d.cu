#include <xfdtd_cuda/calculation_param/calculation_param.cuh>

#include "updator/basic_updator_3d.cuh"
#include "updator/basic_updator_3d_agency.cuh"
#include "updator/update_scheme.cuh"
#include "xfdtd_cuda/common.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_DEVICE auto BasicUpdator3D::updateH() -> void {
  const auto& node_task = _node_task;
  const auto node_size = node_task.xRange().size() * node_task.yRange().size() *
                         node_task.zRange().size();
  const auto node_range = makeRange(Index{0}, node_size);
  // block
  auto grid_size = (gridDim.x * gridDim.y * gridDim.z);
  auto block_id =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  auto block_range = decomposeRange(node_range, block_id, grid_size);

  const auto nx = node_task.xRange().size();
  const auto ny = node_task.yRange().size();
  const auto nz = node_task.zRange().size();

  update<xfdtd::EMF::Attribute::H, Axis::XYZ::X>(
      *_emf, *_calculation_param->fdtdCoefficient(), block_range.start(),
      block_range.end(), nx, ny, nz);
  update<xfdtd::EMF::Attribute::H, Axis::XYZ::Y>(
      *_emf, *_calculation_param->fdtdCoefficient(), block_range.start(),
      block_range.end(), nx, ny, nz);
  update<xfdtd::EMF::Attribute::H, Axis::XYZ::Z>(
      *_emf, *_calculation_param->fdtdCoefficient(), block_range.start(),
      block_range.end(), nx, ny, nz);
}

XFDTD_CUDA_DEVICE auto BasicUpdator3D::updateE() -> void {
  const auto& node_task = _node_task;
  const auto node_size = node_task.xRange().size() * node_task.yRange().size() *
                         node_task.zRange().size();
  const auto node_range = makeRange(Index{0}, node_size);
  // block
  auto grid_size = (gridDim.x * gridDim.y * gridDim.z);
  auto block_id =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  auto block_range = decomposeRange(node_range, block_id, grid_size);

  const auto nx = node_task.xRange().size();
  const auto ny = node_task.yRange().size();
  const auto nz = node_task.zRange().size();

  update<xfdtd::EMF::Attribute::E, Axis::XYZ::X>(
      *_emf, *_calculation_param->fdtdCoefficient(), block_range.start(),
      block_range.end(), nx, ny, nz);
  update<xfdtd::EMF::Attribute::E, Axis::XYZ::Y>(
      *_emf, *_calculation_param->fdtdCoefficient(), block_range.start(),
      block_range.end(), nx, ny, nz);
  update<xfdtd::EMF::Attribute::E, Axis::XYZ::Z>(
      *_emf, *_calculation_param->fdtdCoefficient(), block_range.start(),
      block_range.end(), nx, ny, nz);
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
