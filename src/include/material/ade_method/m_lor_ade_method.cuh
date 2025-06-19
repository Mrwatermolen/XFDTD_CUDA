#ifndef __XFDTD_CUDA_M_LOR_ADE_METHOD_CUH__
#define __XFDTD_CUDA_M_LOR_ADE_METHOD_CUH__

#include "material/ade_method/ade_method.cuh"

namespace xfdtd::cuda {

class MLorentzADEMethodStorage : public ADEMethodStorage {
 public:
  using ADEMethodStorage::ADEMethodStorage;
};

}  // namespace xfdtd::cuda

#endif // __XFDTD_CUDA_M_LOR_ADE_METHOD_CUH__
