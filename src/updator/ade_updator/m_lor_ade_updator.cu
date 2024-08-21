#include <xfdtd_cuda/common.cuh>

#include "material/ade_method/ade_method.cuh"
#include "updator/ade_updator/m_lor_ade_updator.cuh"
#include "updator/ade_updator/m_lor_ade_updator_agency.cuh"
#include "updator/ade_updator/template_ade_update_scheme.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_DEVICE auto MLorentzADEUpdator::updateE() -> void {
  const auto block_range = this->blockRange();

  const auto& task = this->task();
  const auto nx = task.xRange().size();
  const auto ny = task.yRange().size();
  const auto nz = task.zRange().size();

  using UpdatorType = MLorentzADEUpdator;
  TemplateADEUpdateScheme::updateE<UpdatorType, xfdtd::Axis::XYZ::X, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
  TemplateADEUpdateScheme::updateE<UpdatorType, xfdtd::Axis::XYZ::Y, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
  TemplateADEUpdateScheme::updateE<UpdatorType, xfdtd::Axis::XYZ::Z, Index>(
      this, block_range.start(), block_range.end(), nx, ny, nz);
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto MLorentzADEUpdator::updateJ(Index i, Index j, Index k,
                                                   const Real e_next,
                                                   const Real e_cur) -> void {
  const auto num_p = this->storage()->numPole();

  const auto& coeff_j_e_n = this->storage()->coeffJENext();
  const auto& coeff_j_e = this->storage()->coeffJE();
  const auto& coeff_j_e_p = this->storage()->coeffJEPrev();
  const auto& coeff_j_j = this->storage()->coeffJJ();
  const auto& coeff_j_j_p = this->storage()->coeffJJPrev();

  const auto e_prev = this->storage()->ePrevious<xyz>()(i, j, k);
  auto& j_arr = this->storage()->jArr<xyz>();
  auto& j_prev_arr = this->storage()->jPrevArr<xyz>();

  for (Index p{0}; p < num_p; ++p) {
    auto j_next = coeff_j_e_n(i, j, k, p) * e_next +
                  coeff_j_e(i, j, k, p) * e_cur +
                  coeff_j_e_p(i, j, k, p) * e_prev +
                  coeff_j_j(i, j, k, p) * j_arr(i, j, k, p) +
                  coeff_j_j_p(i, j, k, p) * j_prev_arr(i, j, k, p);
    j_prev_arr(i, j, k, p) = j_arr(i, j, k, p);
    j_arr(i, j, k, p) = j_next;
  }
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto MLorentzADEUpdator::recordEPrevious(Real e, Index i,
                                                           Index j,
                                                           Index k) -> void {
  this->storage()->ePrevious<xyz>()(i, j, k) = e;
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto MLorentzADEUpdator::calculateJSum(Index i, Index j,
                                                         Index k) -> Real {
  const auto num_p = this->storage()->numPole();
  const auto& coeff_j_sum_j = this->storage()->coeffJSumJ();
  const auto& coeff_j_sum_j_p = this->storage()->coeffJSumJPrev();
  auto& j_arr = this->storage()->jArr<xyz>();
  auto& j_prev_arr = this->storage()->jPrevArr<xyz>();

  Real sum = 0;
  for (Index p{0}; p < this->storage()->numPole(); ++p) {
    sum += coeff_j_sum_j(i, j, k, p) * j_arr(i, j, k, p) +
           coeff_j_sum_j_p(i, j, k, p) * j_prev_arr(i, j, k, p);
  }

  return sum;
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto MLorentzADEUpdator::ePrevious(Index i, Index j,
                                                     Index k) const -> Real {
  return this->storage()->ePrevious<xyz>()(i, j, k);
}

XFDTD_CUDA_DEVICE auto MLorentzADEUpdator::coeffEPrev(Index i, Index j,
                                                      Index k) const -> Real {
  return this->storage()->coeffEEPrev()(i, j, k);
}

XFDTD_CUDA_GLOBAL void kernelCallMLorentzADEUpdatorUpdateE(
    MLorentzADEUpdator* updator) {
  updator->updateE();
}

XFDTD_CUDA_GLOBAL void kernelCallMLorentzADEUpdatorUpdateH(
    MLorentzADEUpdator* updator) {
  updator->updateH();
}

// Agency
auto MLorentzADEUpdatorAgency::updateE(dim3 grid_dim, dim3 block_dim) -> void {
  kernelCallMLorentzADEUpdatorUpdateE<<<grid_dim, block_dim>>>(_updator);
}

auto MLorentzADEUpdatorAgency::updateH(dim3 grid_dim, dim3 block_dim) -> void {
  kernelCallMLorentzADEUpdatorUpdateH<<<grid_dim, block_dim>>>(_updator);
}

}  // namespace xfdtd::cuda
