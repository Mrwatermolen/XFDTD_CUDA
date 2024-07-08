#ifndef __XFDTD_CUDA_PML_CORRECTOR_AGENCY_CUH__
#define __XFDTD_CUDA_PML_CORRECTOR_AGENCY_CUH__

#include <xfdtd/coordinate_system/coordinate_system.h>

#include "corrector/corrector_agency.cuh"

namespace xfdtd::cuda {

template <xfdtd::Axis::XYZ xyz>
class PMLCorrector;

template <xfdtd::Axis::XYZ xyz>
class PMLCorrectorAgency : public CorrectorAgency {
 public:
  PMLCorrectorAgency(PMLCorrector<xyz>* pml_corrector);

  auto correctE(dim3 grid_size, dim3 block_size) -> void override;

  auto correctH(dim3 grid_size, dim3 block_size) -> void override;

 private:
  PMLCorrector<xyz>* _pml_corrector;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_PML_CORRECTOR_AGENCY_CUH__