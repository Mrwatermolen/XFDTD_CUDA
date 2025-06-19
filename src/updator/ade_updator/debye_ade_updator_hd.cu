#include <utility>

#include "material/ade_method/debye_ade_method_hd.cuh"
#include "updator/ade_updator/debye_ade_updator_agency.cuh"
#include "updator/ade_updator/debye_ade_updator_hd.cuh"

namespace xfdtd::cuda {

DebyeADEUpdatorHD::DebyeADEUpdatorHD(
    IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
    std::shared_ptr<CalculationParamHD> calculation_param_hd,
    std::shared_ptr<EMFHD> emf_hd,
    std::shared_ptr<DebyeADEMethodStorageHD> storage_hd)
    : ADEUpdatorHD{task, std::move(grid_space_hd),
                   std::move(calculation_param_hd), std::move(emf_hd),
                   std::move(storage_hd)},
      _agency{nullptr} {
  static_assert(sizeof(DebyeADEUpdator) == sizeof(ADEUpdator),
                "Size of DebyeADEUpdator is not equal to size of ADEUpdator");
}

DebyeADEUpdatorHD::~DebyeADEUpdatorHD() = default;

auto DebyeADEUpdatorHD::copyHostToDevice() -> void {
  auto device =
      Device{task(), gridSpaceHD()->device(), calculationParamHD()->device(),
             emfHD()->device(),
             static_cast<DebyeADEMethodStorage*>(storageHD()->device())};
  this->copyToDevice(&device);
}

auto DebyeADEUpdatorHD::getUpdatorAgency() -> UpdatorAgency* {
  if (_agency == nullptr) {
    if (device() == nullptr) {
      throw std::runtime_error(
          "DebyeADEUpdatorHD::getUpdatorAgency: device is null");
    }

    _agency = std::make_unique<DebeyeADEUpdatorAgency>();

    _agency->setDevice(static_cast<DebyeADEUpdator*>(device()));
  }

  return _agency.get();
}

}  // namespace xfdtd::cuda
