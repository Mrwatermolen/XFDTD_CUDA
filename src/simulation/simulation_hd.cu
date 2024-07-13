#include <xfdtd/boundary/pml.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/grid_space/grid_space.h>
#include <xfdtd/monitor/field_monitor.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf.h>

#include <memory>
#include <vector>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/simulation/simulation_hd.cuh>

#include "boundary/pml_corrector_hd.cuh"
#include "domain/domain_hd.cuh"
#include "monitor/movie_monitor_hd.cuh"
#include "nf2ff/frequency_domain/nf2ff_frequency_domain_hd.cuh"
#include "updator/basic_updator_3d_hd.cuh"
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
  std::unique_ptr<DomainHD> domain_hd = nullptr;

  std::unique_ptr<BasicUpdatorTEHD> updator_te_hd = nullptr;
  std::unique_ptr<BasicUpdator3DHD> updator_3d_hd = nullptr;

  if (_grid_space_hd->host()->dimension() == xfdtd::GridSpace::Dimension::TWO) {
    updator_te_hd = std::make_unique<BasicUpdatorTEHD>(
        task, _grid_space_hd, _calculation_param_hd, _emf_hd);
    updator_te_hd->copyHostToDevice();
    domain_hd = std::make_unique<DomainHD>(
        _grid_dim, _block_dim, _grid_space_hd, _calculation_param_hd, _emf_hd,
        dynamic_cast<UpdatorAgency*>(updator_te_hd->getUpdatorAgency()));
  } else {
    updator_3d_hd = std::make_unique<BasicUpdator3DHD>(
        task, _grid_space_hd, _calculation_param_hd, _emf_hd);
    updator_3d_hd->copyHostToDevice();
    domain_hd = std::make_unique<DomainHD>(
        _grid_dim, _block_dim, _grid_space_hd, _calculation_param_hd, _emf_hd,
        dynamic_cast<UpdatorAgency*>(updator_3d_hd->getUpdatorAgency()));
  }

  // TFSF
  addTFSFCorrectorHD(tfsf_hd);
  for (auto&& tfsf : tfsf_hd) {
    tfsf->copyHostToDevice();
    // domain_hd->addCorrector(tfsf->getTFSFCorrector2DAgency());
    if (_grid_space_hd->host()->dimension() ==
        xfdtd::GridSpace::Dimension::THREE) {
      domain_hd->addCorrector(tfsf->getTFSFCorrector3DAgency());
    } else if (_grid_space_hd->host()->dimension() ==
               xfdtd::GridSpace::Dimension::TWO) {
      domain_hd->addCorrector(tfsf->getTFSFCorrector2DAgency());
    } else {
      throw std::runtime_error("Invalid dimension");
    }
  }

  // PML
  std::vector<std::unique_ptr<PMLCorrectorHD<xfdtd::Axis::XYZ::X>>>
      pml_corrector_x_hd;
  std::vector<std::unique_ptr<PMLCorrectorHD<xfdtd::Axis::XYZ::Y>>>
      pml_corrector_y_hd;
  std::vector<std::unique_ptr<PMLCorrectorHD<xfdtd::Axis::XYZ::Z>>>
      pml_corrector_z_hd;
  addPMLBoundaryHD(pml_corrector_x_hd);
  addPMLBoundaryHD(pml_corrector_y_hd);
  addPMLBoundaryHD(pml_corrector_z_hd);
  for (auto&& pml : pml_corrector_x_hd) {
    pml->copyHostToDevice();
    domain_hd->addCorrector(pml->getAgency());
  }
  for (auto&& pml : pml_corrector_y_hd) {
    pml->copyHostToDevice();
    domain_hd->addCorrector(pml->getAgency());
  }
  for (auto&& pml : pml_corrector_z_hd) {
    pml->copyHostToDevice();
    domain_hd->addCorrector(pml->getAgency());
  }

  // Monitor
  std::vector<std::unique_ptr<MovieMonitorHD<xfdtd::EMF::Field::EX>>>
      movie_mointor_ex_hd;
  std::vector<std::unique_ptr<MovieMonitorHD<xfdtd::EMF::Field::EY>>>
      movie_mointor_ey_hd;
  std::vector<std::unique_ptr<MovieMonitorHD<xfdtd::EMF::Field::EZ>>>
      movie_mointor_ez_hd;

  for (auto&& m : host()->monitors()) {
    auto movie = dynamic_cast<xfdtd::MovieMonitor*>(m.get());
    if (movie == nullptr) {
      continue;
    }

    auto f = dynamic_cast<xfdtd::FieldMonitor*>(movie->frame().get());
    if (f == nullptr) {
      continue;
    }

    if (f->field() == xfdtd::EMF::Field::EX) {
      movie_mointor_ex_hd.emplace_back(
          std::make_unique<MovieMonitorHD<xfdtd::EMF::Field::EX>>(movie,
                                                                  _emf_hd));
      movie_mointor_ex_hd.back()->copyHostToDevice();
      domain_hd->addMonitor(movie_mointor_ex_hd.back()->getAgency());
      continue;
    }

    if (f->field() == xfdtd::EMF::Field::EY) {
      movie_mointor_ey_hd.emplace_back(
          std::make_unique<MovieMonitorHD<xfdtd::EMF::Field::EY>>(movie,
                                                                  _emf_hd));
      movie_mointor_ey_hd.back()->copyHostToDevice();
      domain_hd->addMonitor(movie_mointor_ey_hd.back()->getAgency());
      continue;
    }

    if (f->field() == xfdtd::EMF::Field::EZ) {
      movie_mointor_ez_hd.emplace_back(
          std::make_unique<MovieMonitorHD<xfdtd::EMF::Field::EZ>>(movie,
                                                                  _emf_hd));
      movie_mointor_ez_hd.back()->copyHostToDevice();
      domain_hd->addMonitor(movie_mointor_ez_hd.back()->getAgency());
      continue;
    }
  }

  // NF2FF
  auto nf2ff_fd_hd = getNF2FFFD();
  for (auto&& nf2ff : nf2ff_fd_hd) {
    nf2ff->copyHostToDevice();
    for (auto&& agency : nf2ff->agencies()) {
      domain_hd->addNF2FFFrequencyDomainAgency(agency);
    }
  }


  std::cout << "SimulationHD::run() - domain created \n";

  domain_hd->run();
  std::cout << "SimulationHD::run() - domain run \n";
  copyDeviceToHost();
  std::cout << "SimulationHD::run() - copyDeviceToHost \n";
  for (auto&& m : movie_mointor_ex_hd) {
    m->copyDeviceToHost();
    m->output();
  }
  for (auto&& m : movie_mointor_ey_hd) {
    m->copyDeviceToHost();
    m->output();
  }
  for (auto&& m : movie_mointor_ez_hd) {
    m->copyDeviceToHost();
    m->output();
  }
  for (auto&& n : nf2ff_fd_hd) {
    n->copyDeviceToHost();
  } 
  std::cout << "SimulationHD::run() End!\n";
  {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";
    }
  }
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

    auto hd = std::make_unique<TFSFCorrectorHD>(
        tfsf, _calculation_param_hd->device(), _emf_hd->device());
    tfsf_hd.emplace_back(std::move(hd));
  }
}

template <xfdtd::Axis::XYZ xyz>
auto SimulationHD::addPMLBoundaryHD(
    std::vector<std::unique_ptr<PMLCorrectorHD<xyz>>>& pml_corrector_hd)
    -> void {
  auto boundaries = host()->boundaries();
  if (boundaries.size() == 0) {
    return;
  }

  for (auto&& b : boundaries) {
    auto pml = dynamic_cast<xfdtd::PML*>(b.get());
    if (pml == nullptr) {
      continue;
    }

    if (pml->mainAxis() != xyz) {
      continue;
    }

    auto pml_corrector = std::make_unique<PMLCorrectorHD<xyz>>(pml, _emf_hd);
    pml_corrector_hd.emplace_back(std::move(pml_corrector));
  }
}

auto SimulationHD::getNF2FFFD()
    -> std::vector<std::unique_ptr<NF2FFFrequencyDomainHD>> {
  auto nf2ffs = host()->nf2ffs();
  if (nf2ffs.size() == 0) {
    return {};
  }

  std::vector<std::unique_ptr<NF2FFFrequencyDomainHD>> nf2ff_fd_hd;
  for (const auto& n : nf2ffs) {
    auto fd = dynamic_cast<xfdtd::NFFFTFrequencyDomain*>(n.get());
    if (fd == nullptr) {
      continue;
    }

    nf2ff_fd_hd.emplace_back(std::make_unique<NF2FFFrequencyDomainHD>(
        fd, _grid_space_hd, _calculation_param_hd, _emf_hd));
  }

  return nf2ff_fd_hd;
}

}  // namespace xfdtd::cuda
