#include <xfdtd_cuda/common.cuh>

#include "updator/basic_updator/basic_updator_te.cuh"
#include "updator/basic_updator/basic_updator_te_agency.cuh"
#include "updator/update_scheme.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_DEVICE auto BasicUpdatorTE::updateE() -> void {
  auto block_range = blockRange();

  const auto task = this->task();
  const auto nx = task.xRange().size();
  const auto ny = task.yRange().size();
  const auto nz = task.zRange().size();

  update<xfdtd::EMF::Attribute::E, Axis::XYZ::Z>(
      *_emf, *_calculation_param->fdtdCoefficient(), block_range.start(),
      block_range.end(), nx, ny, nz);
}

XFDTD_CUDA_GLOBAL auto kernelCallBasicUpdatorTeAgencyUpdateH(
    xfdtd::cuda::BasicUpdatorTE* updator) -> void {
  updator->updateH();
}

XFDTD_CUDA_GLOBAL auto kernelCallBasicUpdatorTeAgencyUpdateE(
    xfdtd::cuda::BasicUpdatorTE* updator) -> void {
  updator->updateE();
}

auto BasicUpdatorTEAgency::updateH(dim3 grid_size, dim3 block_size) -> void {
  kernelCallBasicUpdatorTeAgencyUpdateH<<<grid_size, block_size>>>(_updator);
}

auto BasicUpdatorTEAgency::updateE(dim3 grid_size, dim3 block_size) -> void {
  kernelCallBasicUpdatorTeAgencyUpdateE<<<grid_size, block_size>>>(_updator);
}

auto BasicUpdatorTEAgency::setDevice(BasicUpdatorTE* updator) -> void {
  _updator = updator;
}

}  // namespace xfdtd::cuda
