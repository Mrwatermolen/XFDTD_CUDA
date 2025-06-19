#ifndef __XFDTD_CUDA_BASIC_UPDATOR_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_CUH__

#include <xfdtd_cuda/common.cuh>

#include "updator/update_scheme.cuh"
#include "updator/updator.cuh"

namespace xfdtd::cuda {

// can't be instantiated
class BasicUpdator : public Updator {
 public:
  using Updator::Updator;

  XFDTD_CUDA_DEVICE auto updateH() -> void {
    auto block_range = blockRange();
    const auto& task = this->task();
    const auto nx = task.xRange().size();
    const auto ny = task.yRange().size();
    const auto nz = task.zRange().size();

    update<xfdtd::EMF::Attribute::H, Axis::XYZ::X>(
        *emf(), *calculationParam()->fdtdCoefficient(), block_range.start(),
        block_range.end(), nx, ny, nz);
    update<xfdtd::EMF::Attribute::H, Axis::XYZ::Y>(
        *emf(), *calculationParam()->fdtdCoefficient(), block_range.start(),
        block_range.end(), nx, ny, nz);
    update<xfdtd::EMF::Attribute::H, Axis::XYZ::Z>(
        *emf(), *calculationParam()->fdtdCoefficient(), block_range.start(),
        block_range.end(), nx, ny, nz);
  }

 private:
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_CUH__
