#ifndef __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__

#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/index_task.cuh>

namespace xfdtd::cuda {

class EMFHD;
class GridSpaceHD;
class CalculationParamHD;

class BasicUpdatorTE;
class BasicUpdatorTEAgency;
class UpdatorAgency;

class BasicUpdatorTEHD
    : public HostDeviceCarrier<void, xfdtd::cuda::BasicUpdatorTE> {
  using Host = void;
  using Device = xfdtd::cuda::BasicUpdatorTE;

 public:
  BasicUpdatorTEHD(IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
                   std::shared_ptr<CalculationParamHD> calculation_param_hd,
                   std::shared_ptr<EMFHD> emf_hd);

  ~BasicUpdatorTEHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto getUpdatorAgency() -> UpdatorAgency *;

 private:
  IndexTask _task;
  std::shared_ptr<GridSpaceHD> _grid_space_hd;
  std::shared_ptr<CalculationParamHD> _calculation_param_hd;
  std::shared_ptr<EMFHD> _emf_hd;
  std::unique_ptr<BasicUpdatorTEAgency> _updator_agency;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__
