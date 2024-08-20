#ifndef __XFDTD_CUDA_DEBYE_ADE_UPDATOR_HD_CUH__
#define __XFDTD_CUDA_DEBYE_ADE_UPDATOR_HD_CUH__

#include <memory>

#include "material/ade_method/debye_ade_method_hd.cuh"
#include "updator/ade_updator/ade_updator_hd.cuh"
#include "updator/ade_updator/debye_ade_updator.cuh"

namespace xfdtd::cuda {

class DebeyeADEUpdatorAgency;

class DebyeADEUpdatorHD : public ADEUpdatorHD {
  using Host = void;
  using Device = DebyeADEUpdator;

 public:
  DebyeADEUpdatorHD(IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
                    std::shared_ptr<CalculationParamHD> calculation_param_hd,
                    std::shared_ptr<EMFHD> emf_hd,
                    std::shared_ptr<DebyeADEMethodStorageHD> storage_hd);

  ~DebyeADEUpdatorHD() override;

  auto copyHostToDevice() -> void override;

  auto getUpdatorAgency() -> UpdatorAgency* override;

 private:
  std::unique_ptr<DebeyeADEUpdatorAgency> _agency;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DEBYE_ADE_UPDATOR_HD_CUH__
