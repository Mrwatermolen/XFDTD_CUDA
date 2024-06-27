#ifndef __XFDTD_CUDA_PML_CORRECTOR_CUH__
#define __XFDTD_CUDA_PML_CORRECTOR_CUH__

#include "xfdtd/common/type_define.h"
#include "xfdtd/coordinate_system/coordinate_system.h"
#include "xfdtd/cuda/common.cuh"
#include "xfdtd/cuda/tensor.cuh"

namespace xfdtd {

namespace cuda {

template <Axis::XYZ xyz>
class PMLCorrector {
 public:
  auto correctE() -> void;

  auto correctH() -> void;

 private:
  Index _pml_global_e_start, _pml_global_h_start;
  Index _pml_node_e_start, _pml_node_h_start;
  Index _offset_c;
  Index _a_s, _a_n;
  Index _b_s, _b_n;
  Index _c_e_s, _c_e_n;
  Index _c_h_s, _c_h_n;

  xfdtd::cuda::Array1D<Real>*_coeff_a_e, *_coeff_b_e, *_coeff_a_h, *_coeff_b_h;
  xfdtd::cuda::Array3D<Real>*_c_ea_psi_hb, *_c_eb_psi_ha, *_c_ha_psi_eb,
      *_c_hb_psi_ea;
  xfdtd::cuda::Array3D<Real>*_ea_psi_hb, *_eb_psi_ha, *_ha_psi_eb, *_hb_psi_ea;
  xfdtd::cuda::Array3D<Real>*_ea, *_eb, *_ha, *_hb;
};

template <typename T>
inline auto correctPML(T& field, T& psi, const T& coeff_a, const T& coeff_b,
                       const T& field_p, const T& field_q, const T& c_psi) {
  psi = coeff_b * psi + coeff_a * (field_p - field_q);
  field += c_psi * psi;
}

template <Axis::XYZ xyz>
auto PMLCorrector<xyz>::correctE() -> void {
  // if constexpr (xyz == Axis::XYZ::X) {
  //   const auto c_start = _c_e_s;
  //   const auto c_end = _c_e_s + _c_e_n;
  //   for (Index c_i{c_start}; c_i < c_end; ++c_i) {
  //     auto i = c_i - _pml_node_e_start;
  //     auto global_i = c_i + _offset_c;

  //     for (Index node_j{_a_s}; node_j < _a_s + _a_n; ++node_j) {
  //       for (Index node_k{_b_s}; node_k < _b_s + _b_n + 1; ++node_k) {
  //         auto&& field_v = (*_ea)(c_i, node_j, node_k);
  //         auto&& psi_v = (*_ea_psi_hb)(i, node_j, node_k);
  //         const auto coeff_a_v = (*_coeff_a_e)(global_i - _pml_global_e_start);
  //         const auto coeff_b_v = (*_coeff_b_e)(global_i - _pml_global_e_start);
  //         const auto field_p_v = _hb(c_i, node_j, node_k);
  //         const auto field_q_v = _hb(c_i, node_j - 1, node_k);
  //         const auto c_psi_v = _c_ea_psi_hb(i, node_j, node_k);
  //         correctPML(field_v, psi_v, coeff_a_v, coeff_b_v, field_p_v, field_q_v,
  //                    c_psi_v);
  //       }
  //     }

  //     for (std::size_t node_j{_a_s}; node_j < _a_s + _a_n + 1; ++node_j) {
  //       for (std::size_t node_k{_b_s}; node_k < _b_s + _b_n; ++node_k) {
  //         correctPML(_eb(c_i, node_j, node_k), _eb_psi_ha(i, node_j, node_k),
  //                    _coeff_a_e(global_i - _pml_global_e_start),
  //                    _coeff_b_e(global_i - _pml_global_e_start),
  //                    _ha(c_i, node_j, node_k), _ha(c_i - 1, node_j, node_k),
  //                    _c_eb_psi_ha(i, node_j, node_k));
  //       }
  //     }
  //   }
  // }
}

}  // namespace cuda

}  // namespace xfdtd

#endif  // __XFDTD_CUDA_PML_CORRECTOR_CUH__
