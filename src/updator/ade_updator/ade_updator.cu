#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/common.cuh>

#include "updator/ade_updator/ade_updator.cuh"
#include "updator/ade_updator/drude_ade_updator.cuh"
#include "updator/ade_updator/drude_ade_updator_agency.cuh"
#include "updator/ade_updator/template_ade_update_scheme.cuh"
#include "updator/update_scheme.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_DEVICE auto ADEUpdator::blockRange() const -> IndexRange {
  const auto& node_task = nodeTask();
  const auto node_size = node_task.xRange().size() * node_task.yRange().size() *
                         node_task.zRange().size();
  const auto node_range = makeRange(Index{0}, node_size);
  // block
  auto grid_size = (gridDim.x * gridDim.y * gridDim.z);
  auto block_id =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  return decomposeRange(node_range, block_id, grid_size);
}

XFDTD_CUDA_DEVICE auto ADEUpdator::updateH() -> void {
  auto block_range = blockRange();
  const auto& node_task = nodeTask();
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

// Why do we separate compliation of DrudeADEUpdator from ADEUpdator?
// Because CUDA global function.
// [Separate Compilation and Linking of CUDA C++ Device
// Code](https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code)

// ===========================================================DrudeADEUpdator===========================================================

XFDTD_CUDA_DEVICE auto DrudeADEUpdator::updateE() -> void {
  const auto block_range = blockRange();

  const auto& node_task = nodeTask();
  const auto nx = node_task.xRange().size();
  const auto ny = node_task.yRange().size();
  const auto nz = node_task.zRange().size();

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
  updator->ADEUpdator::updateH();
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

// ===========================================================DrudeADEUpdator===========================================================

}  // namespace xfdtd::cuda
