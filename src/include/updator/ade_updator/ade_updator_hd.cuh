#ifndef __XFDTD_CUDA_ADE_UPDATOR_HD_CUH__
#define __XFDTD_CUDA_ADE_UPDATOR_HD_CUH__

#include <memory>
#include <xfdtd_cuda/host_device_carrier.cuh>

#include "updator/ade_updator/ade_updator.cuh"
#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

class CalculationParamHD;
class EMFHD;
class GridSpaceHD;
class ADEMethodStorageHD;

class ADEUpdatorHD : public HostDeviceCarrier<void, ADEUpdator> {
  using Host = void;
  using Device = ADEUpdator;

 public:
  ADEUpdatorHD(IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
               std::shared_ptr<CalculationParamHD> calculation_param_hd,
               std::shared_ptr<EMFHD> emf_hd,
               std::shared_ptr<ADEMethodStorageHD> storage_hd);

  ~ADEUpdatorHD() override;

  virtual auto getUpdatorAgency() -> UpdatorAgency* = 0;

 protected:
  IndexTask _task;
  std::shared_ptr<GridSpaceHD> _grid_space_hd;
  std::shared_ptr<CalculationParamHD> _calculation_param_hd;
  std::shared_ptr<EMFHD> _emf_hd;
  std::shared_ptr<ADEMethodStorageHD> _storage_hd;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_ADE_UPDATOR_HD_CUH__
