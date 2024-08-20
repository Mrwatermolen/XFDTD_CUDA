#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/index_task.cuh>

#include "material/ade_method/drude_ade_method_hd.cuh"
#include "updator/ade_updator/ade_updator_hd.cuh"
#include "updator/ade_updator/drude_ade_updator_agency.cuh"
#include "updator/ade_updator/drude_ade_updator_hd.cuh"

namespace xfdtd::cuda {

DrudeADEUpdatorHD::DrudeADEUpdatorHD(
    IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
    std::shared_ptr<CalculationParamHD> calculation_param_hd,
    std::shared_ptr<EMFHD> emf_hd,
    std::shared_ptr<DrudeADEMethodStorageHD> storage_hd)
    : ADEUpdatorHD{task, std::move(grid_space_hd),
                   std::move(calculation_param_hd), std::move(emf_hd),
                   std::move(storage_hd)} {}

DrudeADEUpdatorHD::~DrudeADEUpdatorHD() = default;

auto DrudeADEUpdatorHD::copyHostToDevice() -> void {
  auto device = Device{};
  device._node_task = _task;
  device._calculation_param = _calculation_param_hd->device();
  device._emf = _emf_hd->device();
  device._ade_method_storage = _storage_hd->device();
  this->copyToDevice(&device);
}

auto DrudeADEUpdatorHD::copyDeviceToHost() -> void {}

auto DrudeADEUpdatorHD::releaseDevice() -> void {}

auto DrudeADEUpdatorHD::getUpdatorAgency() -> UpdatorAgency* {
  if (_agency == nullptr) {
    if (device() == nullptr) {
      throw std::runtime_error(
          "DrudeADEUpdatorHD::getUpdatorAgency: device is null");
    }

    _agency = std::make_unique<DrudeADEUpdatorAgency>();

    _agency->setDevice(static_cast<DrudeADEUpdator*>(device()));
  }

  return _agency.get();
}

}  // namespace xfdtd::cuda
