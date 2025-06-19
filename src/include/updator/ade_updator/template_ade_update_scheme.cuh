#ifndef __XFDTD_CUDA_TEMPLATE_ADE_UPDATE_SCHEME_CUH__
#define __XFDTD_CUDA_TEMPLATE_ADE_UPDATE_SCHEME_CUH__

#include <xfdtd/util/transform/abc_xyz.h>

#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>

#include "material/ade_method/ade_method.cuh"
#include "updator/update_scheme.cuh"

namespace xfdtd::cuda {

class TemplateADEUpdateScheme {
 public:
  template <typename ADEUpdator, xfdtd::Axis::XYZ xyz, typename Size>
  XFDTD_CUDA_DEVICE static auto updateE(ADEUpdator* ade_updator,
                                        const Size start, const Size end,
                                        const Size nx, const Size ny,
                                        const Size nz) -> void {
    const auto& update_coefficient =
        ade_updator->calculationParam()->fdtdCoefficient();
    auto&& emf = ade_updator->emf();

    constexpr auto attribute = xfdtd::EMF::Attribute::E;
    constexpr auto dual_attribute = xfdtd::EMF::Attribute::H;
    constexpr auto xzy_a = xfdtd::Axis::tangentialAAxis<xyz>();
    constexpr auto xzy_b = xfdtd::Axis::tangentialBAxis<xyz>();

    const auto& cfcf = update_coefficient->template coeff<attribute, xyz>();
    const auto& cf_a =
        update_coefficient
            ->template coeff<attribute, xyz, dual_attribute, xzy_a>();
    const auto& cf_b =
        update_coefficient
            ->template coeff<attribute, xyz, dual_attribute, xzy_b>();

    auto&& field = emf->template field<attribute, xyz>();
    const auto& field_a = emf->template field<dual_attribute, xzy_a>();
    const auto& field_b = emf->template field<dual_attribute, xzy_b>();

    auto&& storage = ade_updator->storage();

    const auto& coeff_e_j_sum = storage->coeffEJSum();
    auto& field_prev = storage->template ePrevious<xyz>();

    const auto thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                           threadIdx.z * blockDim.x * blockDim.y;
    const auto num_threads = blockDim.x * blockDim.y * blockDim.z;

    for (auto index = start; index < end; index += num_threads) {
      if (end <= index + thread_id) {
        break;
      }

      auto [i, j, k] = ijk(index + thread_id, nx, ny, nz);
      auto [a, b, c] = transform::xYZToABC<Index, xyz>(i, j, k);
      auto b_1 = b - 1;
      auto a_1 = a - 1;

      auto [i_a, j_a, k_a] = transform::aBCToXYZ<Index, xyz>(a, b_1, c);

      auto [i_b, j_b, k_b] = transform::aBCToXYZ<Index, xyz>(a_1, b, c);

      if (a == 0 || b == 0) {
        continue;
      }

      const auto j_sum = ade_updator->template calculateJSum<xyz>(i, j, k);
      const auto e_prev = ade_updator->template ePrevious<xyz>(i, j, k);
      const auto coeff_e_e_p = ade_updator->coeffEPrev(i, j, k);

      const auto e_cur = field(i, j, k);
      field(i, j, k) =
          coeff_e_e_p * e_prev +
          eNext(cfcf(i, j, k), field(i, j, k), cf_a(i, j, k), field_a(i, j, k),
                field_a(i_a, j_a, k_a), cf_b(i, j, k), field_b(i, j, k),
                field_b(i_b, j_b, k_b)) +
          coeff_e_j_sum(i, j, k) * j_sum;
      ade_updator->template updateJ<xyz>(i, j, k, field(i, j, k), e_cur);

      ade_updator->template recordEPrevious<xyz>(e_cur, i, j, k);
    }
  }
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_TEMPLATE_ADE_UPDATE_SCHEME_CUH__
