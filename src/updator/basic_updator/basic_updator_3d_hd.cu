#include "updator/basic_updator/basic_updator.cuh"
#include "updator/basic_updator/basic_updator_3d.cuh"
#include "updator/basic_updator/basic_updator_3d_agency.cuh"
#include "updator/basic_updator/basic_updator_3d_hd.cuh"

namespace xfdtd::cuda {

BasicUpdator3DHD::~BasicUpdator3DHD() { releaseDevice(); }

auto BasicUpdator3DHD::getUpdatorAgency() -> UpdatorAgency* {
  static_assert(
      sizeof(BasicUpdator3D) == sizeof(BasicUpdator),
      "The size of BasicUpdator3D and BasicUpdator must be the same.");

  if (!_updator_agency) {
    _updator_agency = std::make_unique<BasicUpdator3DAgency>();
    _updator_agency->setDevice(static_cast<BasicUpdator3D*>(this->device()));
  }

  return _updator_agency.get();
}

}  // namespace xfdtd::cuda
