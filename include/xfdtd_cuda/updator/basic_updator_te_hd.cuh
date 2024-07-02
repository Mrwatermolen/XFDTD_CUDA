#ifndef __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__

#include <xfdtd/simulation/simulation.h>

#include <xfdtd_cuda/updator/basic_updator_te.cuh>
#include <xfdtd_cuda/updator/basic_updator_te_agency.cuh>

#include "xfdtd_cuda/calculation_param/calculation_param_hd.cuh"
#include "xfdtd_cuda/common.cuh"
#include "xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh"
#include "xfdtd_cuda/grid_space/grid_space_hd.cuh"
#include "xfdtd_cuda/host_device_carrier.cuh"
#include "xfdtd_cuda/index_task.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_GLOBAL auto __updateH(xfdtd::cuda::BasicUpdatorTE* updator) -> void {
  updator->updateH();
}

XFDTD_CUDA_GLOBAL auto __updateE(xfdtd::cuda::BasicUpdatorTE* updator) -> void {
  updator->updateE();
}

class BasicUpdatorTEHD
    : public HostDeviceCarrier<void, xfdtd::cuda::BasicUpdatorTE> {
  using Host = void;
  using Device = xfdtd::cuda::BasicUpdatorTE;

 public:
  BasicUpdatorTEHD(IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
                   std::shared_ptr<CalculationParamHD> calculation_param_hd,
                   std::shared_ptr<EMFHD> emf_hd)
      : HostDeviceCarrier{nullptr},
        _task{task},
        _grid_space_hd{grid_space_hd},
        _calculation_param_hd{calculation_param_hd},
        _emf_hd{emf_hd},
        _updator_agency(std::make_unique<BasicUpdatorTEAgency>()) {}

  ~BasicUpdatorTEHD() override { releaseDevice(); }

  auto copyHostToDevice() -> void override {
    auto d = Device{};
    d._node_task = _task;
    d._grid_space = _grid_space_hd->device();
    d._calculation_param = _calculation_param_hd->device();
    d._emf = _emf_hd->device();

    copyToDevice(&d);
    _updator_agency->setDevice(device());
  }

  auto copyDeviceToHost() -> void override {
    // do nothing
  }

  auto releaseDevice() -> void override {
    releaseBaseDevice();
    _updator_agency->setDevice(nullptr);
  }

  auto getUpdatorAgency() { return _updator_agency.get(); }

 private:
  IndexTask _task;
  std::shared_ptr<GridSpaceHD> _grid_space_hd;
  std::shared_ptr<CalculationParamHD> _calculation_param_hd;
  std::shared_ptr<EMFHD> _emf_hd;
  std::unique_ptr<BasicUpdatorTEAgency> _updator_agency;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__
