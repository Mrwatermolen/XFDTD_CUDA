#ifndef __XFDTD_CUDA_ADE_METHOD_CUH__
#define __XFDTD_CUDA_ADE_METHOD_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/tensor.cuh>

namespace xfdtd::cuda {

// class ADEMethodStorageHd;

class ADEMethodStorage {
  friend class ADEMethodStorageHd;

 public:
  XFDTD_CUDA_DEVICE auto numPole() const { return _num_pole; }

  XFDTD_CUDA_DEVICE auto& coeffJJ() { return *_coeff_j_j; }

  XFDTD_CUDA_DEVICE auto& coeffJJ() const { return *_coeff_j_j; }

  XFDTD_CUDA_DEVICE auto& coeffJJPrev() { return *_coeff_j_j_p; }

  XFDTD_CUDA_DEVICE auto& coeffJJPrev() const { return *_coeff_j_j_p; }

  XFDTD_CUDA_DEVICE auto& coeffJENext() { return *_coeff_j_e_n; }

  XFDTD_CUDA_DEVICE auto& coeffJENext() const { return *_coeff_j_e_n; }

  XFDTD_CUDA_DEVICE auto& coeffJE() { return *_coeff_j_e; }

  XFDTD_CUDA_DEVICE auto& coeffJE() const { return *_coeff_j_e; }

  XFDTD_CUDA_DEVICE auto& coeffJEPrev() { return *_coeff_j_e_p; }

  XFDTD_CUDA_DEVICE auto& coeffJEPrev() const { return *_coeff_j_e_p; }

  XFDTD_CUDA_DEVICE auto& coeffJSumJ() { return *_coeff_j_sum_j; }

  XFDTD_CUDA_DEVICE auto& coeffJSumJ() const { return *_coeff_j_sum_j; }

  XFDTD_CUDA_DEVICE auto& coeffJSumJPrev() { return *_coeff_j_sum_j_p; }

  XFDTD_CUDA_DEVICE auto& coeffJSumJPrev() const { return *_coeff_j_sum_j_p; }

  XFDTD_CUDA_DEVICE auto& coeffEJSum() { return *_coeff_e_j_sum; }

  XFDTD_CUDA_DEVICE auto& coeffEJSum() const { return *_coeff_e_j_sum; }

  XFDTD_CUDA_DEVICE auto& coeffEEPrev() { return *_coeff_e_e_p; }

  XFDTD_CUDA_DEVICE auto& coeffEEPrev() const { return *_coeff_e_e_p; }

  XFDTD_CUDA_DEVICE auto& exPrev() { return *_ex_prev; }

  XFDTD_CUDA_DEVICE auto& exPrev() const { return *_ex_prev; }

  XFDTD_CUDA_DEVICE auto& eyPrev() { return *_ey_prev; }

  XFDTD_CUDA_DEVICE auto& eyPrev() const { return *_ey_prev; }

  XFDTD_CUDA_DEVICE auto& ezPrev() { return *_ez_prev; }

  XFDTD_CUDA_DEVICE auto& ezPrev() const { return *_ez_prev; }

  XFDTD_CUDA_DEVICE auto& jxArr() { return *_jx_arr; }

  XFDTD_CUDA_DEVICE auto& jxArr() const { return *_jx_arr; }

  XFDTD_CUDA_DEVICE auto& jyArr() { return *_jy_arr; }

  XFDTD_CUDA_DEVICE auto& jyArr() const { return *_jy_arr; }

  XFDTD_CUDA_DEVICE auto& jzArr() { return *_jz_arr; }

  XFDTD_CUDA_DEVICE auto& jzArr() const { return *_jz_arr; }

  XFDTD_CUDA_DEVICE auto& jxPrevArr() { return *_jx_prev_arr; }

  XFDTD_CUDA_DEVICE auto& jxPrevArr() const { return *_jx_prev_arr; }

  XFDTD_CUDA_DEVICE auto& jyPrevArr() { return *_jy_prev_arr; }

  XFDTD_CUDA_DEVICE auto& jyPrevArr() const { return *_jy_prev_arr; }

  XFDTD_CUDA_DEVICE auto& jzPrevArr() { return *_jz_prev_arr; }

  XFDTD_CUDA_DEVICE auto& jzPrevArr() const { return *_jz_prev_arr; }

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto& ePrevious() {
    return const_cast<Array3D<Real>&>(
        static_cast<const ADEMethodStorage*>(this)->ePrevious<xyz>());
  }

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto& ePrevious() const {
    if constexpr (xyz == xfdtd::Axis::XYZ::X) {
      return exPrev();
    } else if constexpr (xyz == xfdtd::Axis::XYZ::Y) {
      return eyPrev();
    } else if constexpr (xyz == xfdtd::Axis::XYZ::Z) {
      return ezPrev();
    }
  }

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto& jArr() {
    return const_cast<Array4D<Real>&>(
        static_cast<const ADEMethodStorage*>(this)->jArr<xyz>());
  }

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto& jArr() const {
    if constexpr (xyz == xfdtd::Axis::XYZ::X) {
      return jxArr();
    } else if constexpr (xyz == xfdtd::Axis::XYZ::Y) {
      return jyArr();
    } else if constexpr (xyz == xfdtd::Axis::XYZ::Z) {
      return jzArr();
    }
  }

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto& jPrevArr() {
    return const_cast<Array4D<Real>&>(
        static_cast<const ADEMethodStorage*>(this)->jPrevArr<xyz>());
  }

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto& jPrevArr() const {
    if constexpr (xyz == xfdtd::Axis::XYZ::X) {
      return jxPrevArr();
    } else if constexpr (xyz == xfdtd::Axis::XYZ::Y) {
      return jyPrevArr();
    } else if constexpr (xyz == xfdtd::Axis::XYZ::Z) {
      return jzPrevArr();
    }
  }

 protected:
  Index _num_pole{};
  Array4D<Real>*_coeff_j_j{}, *_coeff_j_j_p{}, *_coeff_j_e_n{}, *_coeff_j_e{},
      *_coeff_j_e_p{}, *_coeff_j_sum_j{}, *_coeff_j_sum_j_p{};
  Array3D<Real>*_coeff_e_j_sum{}, *_coeff_e_e_p{};
  Array3D<Real>*_ex_prev{}, *_ey_prev{}, *_ez_prev{};
  Array4D<Real>*_jx_arr{}, *_jy_arr{}, *_jz_arr{}, *_jx_prev_arr{},
      *_jy_prev_arr{}, *_jz_prev_arr{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_ADE_METHOD_CUH__
