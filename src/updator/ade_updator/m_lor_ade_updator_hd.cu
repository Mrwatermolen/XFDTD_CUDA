#include "updator/ade_updator/m_lor_ade_updator_agency.cuh"
#include "updator/ade_updator/m_lor_ade_updator_hd.cuh"

namespace xfdtd::cuda {

MLorentzADEUpdatorHD::MLorentzADEUpdatorHD(
    IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
    std::shared_ptr<CalculationParamHD> calculation_param_hd,
    std::shared_ptr<EMFHD> emf_hd,
    std::shared_ptr<MLorentzADEMethodStorageHD> storage_hd)
    : ADEUpdatorHD{task, std::move(grid_space_hd),
                   std::move(calculation_param_hd), std::move(emf_hd),
                   std::move(storage_hd)},
      _agency{nullptr} {}

MLorentzADEUpdatorHD::~MLorentzADEUpdatorHD() = default;

auto MLorentzADEUpdatorHD::copyHostToDevice() -> void {
  auto d =
      Device{task(), gridSpaceHD()->device(), calculationParamHD()->device(),
             emfHD()->device(),
             static_cast<MLorentzADEMethodStorage*>(storageHD()->device())};

  this->copyToDevice(&d);
}

auto MLorentzADEUpdatorHD::getUpdatorAgency() -> UpdatorAgency* {
  if (_agency == nullptr) {
    if (device() == nullptr) {
      throw std::runtime_error(
          "MLorentzADEUpdatorHD::getUpdatorAgency: device is null");
    }

    _agency = std::make_unique<MLorentzADEUpdatorAgency>(
        static_cast<MLorentzADEUpdator*>(device()));
  }

  return _agency.get();
}

}  // namespace xfdtd::cuda
