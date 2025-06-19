#ifndef __XFDTD_CUDA_DRUDE_ADE_METHOD_CUH__
#define __XFDTD_CUDA_DRUDE_ADE_METHOD_CUH__

#include <xfdtd/common/type_define.h>

#include "material/ade_method/ade_method.cuh"

namespace xfdtd::cuda {

class DrudeADEMethodStorage : public ADEMethodStorage {
 public:
  DrudeADEMethodStorage(Index num_pole, Array4D<Real>* coeff_j_j,
                        Array4D<Real>* coeff_j_e, Array4D<Real>* coeff_j_sum_j,
                        Array3D<Real>* coeff_e_j_sum, Array4D<Real>* jx_arr,
                        Array4D<Real>* jy_arr, Array4D<Real>* jz_arr);
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DRUDE_ADE_METHOD_CUH__
