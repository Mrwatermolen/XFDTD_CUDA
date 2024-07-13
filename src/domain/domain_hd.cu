#include <xfdtd/calculation_param/time_param.h>

#include <memory>
#include <vector>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/calculation_param/time_param.cuh>
#include <xfdtd_cuda/calculation_param/time_param_hd.cuh>
#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>

#include "corrector/corrector_agency.cuh"
#include "domain/domain_hd.cuh"
#include "monitor/monitor_agency.cuh"
#include "nf2ff/frequency_domain/nf2ff_frequency_domain_agency.cuh"
#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

DomainHD::DomainHD(dim3 grid_dim, dim3 block_dim,
                   std::shared_ptr<GridSpaceHD> grid_space_hd,
                   std::shared_ptr<CalculationParamHD> calculation_param_hd,
                   std::shared_ptr<EMFHD> emf_hd, UpdatorAgency* updator)
    : _grid_dim{grid_dim},
      _block_dim{block_dim},
      _grid_space_hd{grid_space_hd},
      _calculation_param_hd{calculation_param_hd},
      _emf_hd{emf_hd},
      _updator{updator} {}

DomainHD::~DomainHD() = default;

auto DomainHD::run() -> void {
  while (!isCalculationDone()) {
    updateH();
    correctH();
    updateE();
    correctE();
    record();
    nextStep();
  }
  cudaDeviceSynchronize();
}

auto DomainHD::updateH() -> void { _updator->updateH(_grid_dim, _block_dim); }

auto DomainHD::correctH() -> void {
  for (auto corrector : _correctors) {
    corrector->correctH(_grid_dim, _block_dim);
  }
}

auto DomainHD::updateE() -> void { _updator->updateE(_grid_dim, _block_dim); }

auto DomainHD::correctE() -> void {
  for (auto corrector : _correctors) {
    corrector->correctE(_grid_dim, _block_dim);
  }
}

auto DomainHD::record() -> void {
  for (auto monitor : _monitors) {
    monitor->update(_grid_dim, _block_dim);
  }

  for (auto nf2ff : _nf2ff_agencies) {
    nf2ff->update(_grid_dim, _block_dim);
  }
}

auto DomainHD::nextStep() -> void {
  _calculation_param_hd->timeParamHD()->nextStepInDevice();
  _calculation_param_hd->timeParamHD()->host()->nextStep();
}

auto DomainHD::isCalculationDone() -> bool {
  return _calculation_param_hd->timeParamHD()->host()->endTimeStep() <=
         _calculation_param_hd->timeParamHD()->host()->currentTimeStep();
}

auto DomainHD::addCorrector(CorrectorAgency* corrector) -> void {
  _correctors.push_back(corrector);
}

auto DomainHD::addMonitor(MonitorAgency* monitor) -> void {
  _monitors.push_back(monitor);
}

auto DomainHD::addNF2FFFrequencyDomainAgency(NF2FFFrequencyDomainAgency* agency)
    -> void {
  _nf2ff_agencies.push_back(agency);
}

}  // namespace xfdtd::cuda
