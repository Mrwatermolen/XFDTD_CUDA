#ifndef __XFDTD_CUDA_ADE_METHOD_HD_CUH__
#define __XFDTD_CUDA_ADE_METHOD_HD_CUH__

#include <xfdtd/material/ade_method/ade_method.h>

#include <xfdtd_cuda/host_device_carrier.cuh>

#include "material/ade_method/ade_method.cuh"
#include "xfdtd_cuda/tensor_hd.cuh"

namespace xfdtd::cuda {

class ADEMethodStorageHD
    : public HostDeviceCarrier<xfdtd::ADEMethodStorage,
                               xfdtd::cuda::ADEMethodStorage> {
  using Host = xfdtd::ADEMethodStorage;
  using Device = xfdtd::cuda::ADEMethodStorage;

 public:
  explicit ADEMethodStorageHD(Host* host);

 protected:
  TensorHD<Real, 4> _coeff_j_j_hd, _coeff_j_j_p_hd, _coeff_j_e_n_hd,
      _coeff_j_e_hd, _coeff_j_e_p_hd, _coeff_j_sum_j_hd;
  TensorHD<Real, 3> _coeff_e_j_sum_hd, _coeff_e_e_p_hd;
  TensorHD<Real, 3> _ex_prev_hd, _ey_prev_hd, _ez_prev_hd;
  TensorHD<Real, 4> _jx_arr_hd, _jy_arr_hd, _jz_arr_hd, _jx_prev_arr_hd,
      _jy_prev_arr_hd, _jz_prev_arr_hd;
 private:
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_ADE_METHOD_HD_CUH__
