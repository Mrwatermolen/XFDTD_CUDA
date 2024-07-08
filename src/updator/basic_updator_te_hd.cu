#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>

#include "updator/basic_updator_te.cuh"
#include "updator/basic_updator_te_agency.cuh"
#include "updator/basic_updator_te_hd.cuh"

namespace xfdtd::cuda {

BasicUpdatorTEHD::BasicUpdatorTEHD(
    IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
    std::shared_ptr<CalculationParamHD> calculation_param_hd,
    std::shared_ptr<EMFHD> emf_hd)
    : HostDeviceCarrier{nullptr},
      _task{task},
      _grid_space_hd{grid_space_hd},
      _calculation_param_hd{calculation_param_hd},
      _emf_hd{emf_hd},
      _updator_agency(std::make_unique<BasicUpdatorTEAgency>()) {}

BasicUpdatorTEHD::~BasicUpdatorTEHD() { releaseDevice(); }

auto BasicUpdatorTEHD::copyHostToDevice() -> void {
  auto d = Device{};
  d._node_task = _task;
  d._grid_space = _grid_space_hd->device();
  d._calculation_param = _calculation_param_hd->device();
  d._emf = _emf_hd->device();

  copyToDevice(&d);
  _updator_agency->setDevice(device());
}

auto BasicUpdatorTEHD::copyDeviceToHost() -> void {
  // do nothing
}

auto BasicUpdatorTEHD::releaseDevice() -> void {
  this->releaseBaseDevice();
  _updator_agency->setDevice(nullptr);
}

auto BasicUpdatorTEHD::getUpdatorAgency() -> UpdatorAgency* {
  return _updator_agency.get();
}

}  // namespace xfdtd::cuda
