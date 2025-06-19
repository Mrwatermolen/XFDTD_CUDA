#include <xfdtd/coordinate_system/coordinate_system.h>

#include <xfdtd_cuda/common.cuh>

#include "material/ade_method/ade_method.cuh"
#include "material/ade_method/drude_ade_method.cuh"
#include "updator/ade_updator/drude_ade_updator.cuh"
#include "updator/ade_updator/drude_ade_updator_agency.cuh"
#include "updator/ade_updator/template_ade_update_scheme.cuh"

namespace xfdtd::cuda {

DrudeADEUpdator::DrudeADEUpdator(IndexTask task, GridSpace* grid_space,
                                 CalculationParam* calculation_param, EMF* emf,
                                 DrudeADEMethodStorage* storage)
    : ADEUpdator{task, grid_space, calculation_param, emf,
                 dynamic_cast<ADEMethodStorage*>(storage)} {}

XFDTD_CUDA_DEVICE auto DrudeADEUpdator::updateE() -> void {
  const auto block_range = blockRange();

  const auto& task = this->task();
  const auto nx = task.xRange().size();
  const auto ny = task.yRange().size();
  const auto nz = task.zRange().size();

  TemplateADEUpdateScheme::updateE<DrudeADEUpdator, xfdtd::Axis::XYZ::X, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
  TemplateADEUpdateScheme::updateE<DrudeADEUpdator, xfdtd::Axis::XYZ::Y, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
  TemplateADEUpdateScheme::updateE<DrudeADEUpdator, xfdtd::Axis::XYZ::Z, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto DrudeADEUpdator::updateJ(Index i, Index j, Index k,
                                                Real e_next,
                                                Real e_cur) -> void {
  auto num_p = this->storage()->numPole();
  auto& j_arr = this->storage()->jArr<xyz>();
  const auto& coeff_j_j = this->storage()->coeffJJ();
  const auto& coeff_j_e = this->storage()->coeffJE();
  for (Index p{0}; p < num_p; ++p) {
    j_arr(i, j, k, p) = coeff_j_j(i, j, k, p) * j_arr(i, j, k, p) +
                        coeff_j_e(i, j, k, p) * (e_next + e_cur);
  }
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto DrudeADEUpdator::calculateJSum(Index i, Index j,
                                                      Index k) -> Real {
  auto num_p = this->storage()->numPole();
  const auto& j_arr = this->storage()->jArr<xyz>();
  const auto& coeff_j_sum_j = this->storage()->coeffJSumJ();
  Real sum{0};
  for (Index p{0}; p < num_p; ++p) {
    sum += coeff_j_sum_j(i, j, k, p) * j_arr(i, j, k, p);
  }
  return sum;
}

XFDTD_CUDA_GLOBAL void kernelCallDrudeADEUpdatorUpdateE(
    DrudeADEUpdator* updator) {
  updator->updateE();
}

XFDTD_CUDA_GLOBAL void kernelCallDrudeADEUpdatorUpdateH(
    DrudeADEUpdator* updator) {
  updator->updateH();
}

// Agency
auto DrudeADEUpdatorAgency::updateE(dim3 grid_size, dim3 block_size) -> void {
  kernelCallDrudeADEUpdatorUpdateE<<<grid_size, block_size>>>(_updator);
}

auto DrudeADEUpdatorAgency::updateH(dim3 grid_size, dim3 block_size) -> void {
  kernelCallDrudeADEUpdatorUpdateH<<<grid_size, block_size>>>(_updator);
}

auto DrudeADEUpdatorAgency::setDevice(DrudeADEUpdator* updator) -> void {
  _updator = updator;
}

}  // namespace xfdtd::cuda
