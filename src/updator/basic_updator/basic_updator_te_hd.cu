#include "updator/basic_updator/basic_updator_te.cuh"
#include "updator/basic_updator/basic_updator_te_agency.cuh"
#include "updator/basic_updator/basic_updator_te_hd.cuh"

namespace xfdtd::cuda {

BasicUpdatorTEHD::~BasicUpdatorTEHD() { releaseDevice(); }

// auto BasicUpdatorTEHD::copyHostToDevice() -> void {
//   auto d = Device{};
//   d._node_task = _task;
//   d._grid_space = _grid_space_hd->device();
//   d._calculation_param = _calculation_param_hd->device();
//   d._emf = _emf_hd->device();

//   copyToDevice(&d);
//   _updator_agency->setDevice(device());
// }

// auto BasicUpdatorTEHD::copyDeviceToHost() -> void {
//   // do nothing
// }

// auto BasicUpdatorTEHD::releaseDevice() -> void {
//   this->releaseBaseDevice();
//   _updator_agency->setDevice(nullptr);
// }

auto BasicUpdatorTEHD::getUpdatorAgency() -> UpdatorAgency* {
  if (!_updator_agency) {
    _updator_agency = std::make_unique<BasicUpdatorTEAgency>();
    _updator_agency->setDevice(static_cast<BasicUpdatorTE*>(this->device()));
  }

  return _updator_agency.get();
}

}  // namespace xfdtd::cuda
