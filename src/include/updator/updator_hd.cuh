#ifndef __XFDTD_CUDA_UPDATOR_HD_CUH__
#define __XFDTD_CUDA_UPDATOR_HD_CUH__

#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/index_task.cuh>

#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

template <typename TUpdator>
class UpdatorHD : public HostDeviceCarrier<void, TUpdator> {
 public:
  using Host = void;
  using Device = TUpdator;

  UpdatorHD(IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
            std::shared_ptr<CalculationParamHD> calculation_param_hd,
            std::shared_ptr<EMFHD> emf_hd)
      : HostDeviceCarrier<Host, Device>{nullptr},
        _task{task},
        _grid_space_hd{std::move(grid_space_hd)},
        _calculation_param_hd{std::move(calculation_param_hd)},
        _emf_hd{std::move(emf_hd)} {}

  ~UpdatorHD() override { releaseDevice(); }

  auto task() const { return _task; }

  auto gridSpaceHD() const { return _grid_space_hd; }

  auto calculationParamHD() const { return _calculation_param_hd; }

  auto emfHD() const { return _emf_hd; }

  auto gridSpaceHD() { return _grid_space_hd; }

  auto calculationParamHD() { return _calculation_param_hd; }

  auto emfHD() { return _emf_hd; }

  auto copyDeviceToHost() -> void override {}

  auto releaseDevice() -> void override { this->releaseBaseDevice(); }

  virtual auto getUpdatorAgency() -> UpdatorAgency* = 0;

 private:
  IndexTask _task;
  std::shared_ptr<GridSpaceHD> _grid_space_hd;
  std::shared_ptr<CalculationParamHD> _calculation_param_hd;
  std::shared_ptr<EMFHD> _emf_hd;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_UPDATOR_HD_CUH__
