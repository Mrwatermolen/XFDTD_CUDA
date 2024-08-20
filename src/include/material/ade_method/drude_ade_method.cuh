#ifndef __XFDTD_CUDA_DRUDE_ADE_METHOD_CUH__
#define __XFDTD_CUDA_DRUDE_ADE_METHOD_CUH__

#include <xfdtd/common/type_define.h>

#include "material/ade_method/ade_method.cuh"

namespace xfdtd::cuda {

class DrudeADEMethodStorage : public ADEMethodStorage {
  friend class DrudeADEMethodStorageHD;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DRUDE_ADE_METHOD_CUH__
