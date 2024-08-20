#include <xfdtd_cuda/common.cuh>

#include "updator/ade_updator/debye_ade_updator.cuh"
#include "updator/ade_updator/debye_ade_updator_agency.cuh"
#include "updator/ade_updator/template_ade_update_scheme.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_DEVICE auto DebyeADEUpdator::updateE() -> void {
  const auto block_range = blockRange();

  const auto& task = this->task();
  const auto nx = task.xRange().size();
  const auto ny = task.yRange().size();
  const auto nz = task.zRange().size();

  using UpdatorType = DebyeADEUpdator;

  TemplateADEUpdateScheme::updateE<UpdatorType, xfdtd::Axis::XYZ::X, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
  TemplateADEUpdateScheme::updateE<UpdatorType, xfdtd::Axis::XYZ::Y, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
  TemplateADEUpdateScheme::updateE<UpdatorType, xfdtd::Axis::XYZ::Z, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
}

template <Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto DebyeADEUpdator::updateJ(Index i, Index j, Index k,
                                                const Real e_next,
                                                const Real e_cur) -> void {
  auto& j_arr = this->storage()->jArr<xyz>();
  const auto& coeff_j_j = this->storage()->coeffJJ();
  const auto& coeff_j_e = this->storage()->coeffJE();
  const auto num_p = this->storage()->numPole();
  for (Index p{0}; p < num_p; ++p) {
    j_arr(i, j, k, p) = coeff_j_j(i, j, k, p) * j_arr(i, j, k, p) +
                        coeff_j_e(i, j, k, p) * (e_next - e_cur);
  }
}

template <Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto DebyeADEUpdator::calculateJSum(Index i, Index j,
                                                      Index k) -> Real {
  const auto& j_arr = this->storage()->jArr<xyz>();
  const auto& coeff_j_sum_j = this->storage()->coeffJSumJ();
  const auto num_p = this->storage()->numPole();
  Real sum{0};
  for (Index p{0}; p < num_p; ++p) {
    sum += coeff_j_sum_j(i, j, k, p) * j_arr(i, j, k, p);
  }
  return sum;
}

XFDTD_CUDA_GLOBAL void kernelCallDebyeADEUpdatorUpdateE(
    DebyeADEUpdator* updator) {
  updator->updateE();
}

XFDTD_CUDA_GLOBAL void kernelCallDebyeADEUpdatorUpdateH(
    DebyeADEUpdator* updator) {
  updator->updateH();
}

// Agency
auto DebeyeADEUpdatorAgency::updateE(dim3 grid_size, dim3 block_size) -> void {
  kernelCallDebyeADEUpdatorUpdateE<<<grid_size, block_size>>>(_updator);
}

auto DebeyeADEUpdatorAgency::updateH(dim3 grid_size, dim3 block_size) -> void {
  kernelCallDebyeADEUpdatorUpdateH<<<grid_size, block_size>>>(_updator);
}

auto DebeyeADEUpdatorAgency::setDevice(DebyeADEUpdator* updator) -> void {
  _updator = updator;
}

}  // namespace xfdtd::cuda
