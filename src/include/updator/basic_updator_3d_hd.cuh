#ifndef __XFDTD_CUDA_BASIC_UPDATOR_3D_HD_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_3D_HD_CUH__

#include <memory>
#include <xfdtd_cuda/host_device_carrier.cuh>

#include "xfdtd_cuda/index_task.cuh"

namespace xfdtd::cuda {

class BasicUpdator3D;
class CalculationParamHD;
class EMFHD;
class GridSpaceHD;

class BasicUpdator3DAgency;
class UpdatorAgency;

class BasicUpdator3DHD : public HostDeviceCarrier<void, BasicUpdator3D> {
  using Host = void;
  using Device = BasicUpdator3D;

 public:
  BasicUpdator3DHD(IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
                   std::shared_ptr<CalculationParamHD> calculation_param_hd,
                   std::shared_ptr<EMFHD> emf_hd);

  ~BasicUpdator3DHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto getUpdatorAgency() -> UpdatorAgency *;

 private:
  IndexTask _task;
  std::shared_ptr<GridSpaceHD> _grid_space_hd;
  std::shared_ptr<CalculationParamHD> _calculation_param_hd;
  std::shared_ptr<EMFHD> _emf_hd;
  std::unique_ptr<BasicUpdator3DAgency> _updator_agency;
};

};  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_3D_HD_CUH__
