#include <xfdtd/common/type_define.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf.h>

#include <memory>
#include <vector>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/simulation/simulation_hd.cuh>

#include "domain/domain_hd.cuh"
#include "monitor/movie_monitor_hd.cuh"
#include "updator/basic_updator_te_hd.cuh"
#include "updator/updator_agency.cuh"
#include "waveform_source/tfsf/tfsf_corrector_hd.cuh"

namespace xfdtd::cuda {

SimulationHD::SimulationHD(xfdtd::Simulation* host) : HostDeviceCarrier{host} {}

SimulationHD::~SimulationHD() { releaseDevice(); }

auto SimulationHD::copyHostToDevice() -> void {
  if (this->host() == nullptr) {
    throw std::runtime_error("Host is nullptr");
  }

  _grid_space_hd->copyHostToDevice();
  _calculation_param_hd->copyHostToDevice();
  _emf_hd->copyHostToDevice();
}

auto SimulationHD::copyDeviceToHost() -> void {
  if (host() == nullptr) {
    throw std::runtime_error("Host is nullptr");
  }

  _grid_space_hd->copyDeviceToHost();
  _calculation_param_hd->copyDeviceToHost();
  _emf_hd->copyDeviceToHost();
}

auto SimulationHD::releaseDevice() -> void {
  _grid_space_hd->releaseDevice();
  _calculation_param_hd->releaseDevice();
  _emf_hd->releaseDevice();
}

auto SimulationHD::run(Index time_step) -> void {
  std::vector<std::unique_ptr<TFSFCorrectorHD>> tfsf_hd;
  std::cout << "SimulationHD::run() Start!\n";
  init(time_step);
  copyHostToDevice();
  std::cout << "SimulationHD::run() - copyHostToDevice \n";

  auto task =
      xfdtd::cuda::IndexTask{IndexRange{0, _grid_space_hd->host()->sizeX()},
                             IndexRange{0, _grid_space_hd->host()->sizeY()},
                             IndexRange{0, _grid_space_hd->host()->sizeZ()}};
  auto updator_hd = std::make_unique<BasicUpdatorTEHD>(
      task, _grid_space_hd, _calculation_param_hd, _emf_hd);
  updator_hd->copyHostToDevice();
  auto domain_hd = std::make_unique<DomainHD>(
      _grid_dim, _block_dim, _grid_space_hd, _calculation_param_hd, _emf_hd,
      dynamic_cast<UpdatorAgency*>(updator_hd->getUpdatorAgency()));

  addTFSFCorrectorHD(tfsf_hd);

  for (auto&& tfsf : tfsf_hd) {
    tfsf->copyHostToDevice();
    domain_hd->addCorrector(tfsf->getTFSFCorrector2DAgency());
  }

  std::vector<std::unique_ptr<MovieMonitorHD<xfdtd::EMF::Field::EZ>>>
      movie_mointor_hd;
  for (auto&& m : host()->monitors()) {
    auto movie = dynamic_cast<xfdtd::MovieMonitor*>(m.get());
    if (movie == nullptr) {
      continue;
    }

    // assume that only EZ is monitored
    movie_mointor_hd.emplace_back(
        std::make_unique<MovieMonitorHD<xfdtd::EMF::Field::EZ>>(movie,
                                                                _emf_hd));
    movie_mointor_hd.back()->copyHostToDevice();
    domain_hd->addMonitor(movie_mointor_hd.back()->getAgency());
  }

  std::cout << "SimulationHD::run() - domain created \n";

  domain_hd->run();
  std::cout << "SimulationHD::run() - domain run \n";
  copyDeviceToHost();
  std::cout << "SimulationHD::run() - copyDeviceToHost \n";
  for (auto&& m : movie_mointor_hd) {
    std::cout << "SimulationHD::run() - output \n";
    m->copyDeviceToHost();
    m->output();
  }
  std::cout << "SimulationHD::run() End!\n";
}

auto SimulationHD::init(Index time_step) -> void {
  host()->init(time_step);
  _grid_space_hd = std::make_shared<GridSpaceHD>(host()->gridSpace().get());
  _calculation_param_hd =
      std::make_shared<CalculationParamHD>(host()->calculationParam().get());
  _emf_hd = std::make_shared<EMFHD>(host()->emf().get());
}

auto SimulationHD::setGridDim(dim3 grid_dim) -> void { _grid_dim = grid_dim; }

auto SimulationHD::setBlockDim(dim3 block_dim) -> void {
  _block_dim = block_dim;
}

auto SimulationHD::addTFSFCorrectorHD(
    std::vector<std::unique_ptr<TFSFCorrectorHD>>& tfsf_hd) -> void {
  auto sources = host()->waveformSources();
  if (sources.size() == 0) {
    return;
  }

  for (auto&& w : sources) {
    auto tfsf = dynamic_cast<xfdtd::TFSF*>(w.get());
    if (tfsf == nullptr) {
      continue;
    }

    auto tfsf_2d_hd = std::make_unique<TFSFCorrectorHD>(
        tfsf, _calculation_param_hd->device(), _emf_hd->device());
    tfsf_hd.emplace_back(std::move(tfsf_2d_hd));
  }
}

}  // namespace xfdtd::cuda
