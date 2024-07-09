#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>

#include "updator/basic_updator_3d.cuh"
#include "updator/basic_updator_3d_agency.cuh"
#include "updator/basic_updator_3d_hd.cuh"

namespace xfdtd::cuda {

BasicUpdator3DHD::BasicUpdator3DHD(
    IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
    std::shared_ptr<CalculationParamHD> calculation_param_hd,
    std::shared_ptr<EMFHD> emf_hd)
    : HostDeviceCarrier<Host, Device>{nullptr},
      _task{task},
      _emf_hd{emf_hd},
      _calculation_param_hd{calculation_param_hd} {
  _updator_agency = std::make_unique<BasicUpdator3DAgency>();
}

BasicUpdator3DHD::~BasicUpdator3DHD() { releaseDevice(); }

auto BasicUpdator3DHD::copyHostToDevice() -> void {
  auto d = Device{};
  d._node_task = _task;
  d._emf = _emf_hd->device();
  d._calculation_param = _calculation_param_hd->device();

  copyToDevice(&d);
  _updator_agency->setDevice(device());
}

auto BasicUpdator3DHD::copyDeviceToHost() -> void {
  // do nothing
}

auto BasicUpdator3DHD::releaseDevice() -> void {
  releaseBaseDevice();
  _updator_agency->setDevice(nullptr);
}

auto BasicUpdator3DHD::getUpdatorAgency() -> UpdatorAgency* {
  return _updator_agency.get();
}

}  // namespace xfdtd::cuda
