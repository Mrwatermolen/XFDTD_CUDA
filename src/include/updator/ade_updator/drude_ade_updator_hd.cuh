#ifndef __XFDTD_CUDA_DRUDE_ADE_UPDATOR_HD_CUH__
#define __XFDTD_CUDA_DRUDE_ADE_UPDATOR_HD_CUH__

#include <memory>
#include <xfdtd_cuda/host_device_carrier.cuh>

#include "updator/ade_updator/ade_updator_hd.cuh"
#include "updator/ade_updator/drude_ade_updator.cuh"
#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

class CalculationParamHD;
class EMFHD;
class GridSpaceHD;
class DrudeADEMethodStorageHD;

class DrudeADEUpdatorAgency;

class DrudeADEUpdatorHD : public ADEUpdatorHD {
  using Host = void;
  using Device = DrudeADEUpdator;

 public:
  DrudeADEUpdatorHD(IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
                    std::shared_ptr<CalculationParamHD> calculation_param_hd,
                    std::shared_ptr<EMFHD> emf_hd,
                    std::shared_ptr<DrudeADEMethodStorageHD> storage_hd);

  ~DrudeADEUpdatorHD() override;

  auto copyHostToDevice() -> void override;

  // auto releaseDevice() -> void override;

  auto getUpdatorAgency() -> UpdatorAgency* override;

 private:
  std::unique_ptr<DrudeADEUpdatorAgency> _agency;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DRUDE_ADE_UPDATOR_HD_CUH__
