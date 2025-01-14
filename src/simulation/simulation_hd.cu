#include <xfdtd/boundary/pml.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/grid_space/grid_space.h>
#include <xfdtd/material/ade_method/drude_ade_method.h>
#include <xfdtd/material/ade_method/m_lor_ade_method.h>
#include <xfdtd/monitor/field_monitor.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/nffft/nffft_time_domain.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/simulation/simulation_flag.h>
#include <xfdtd/waveform_source/tfsf.h>

#include <memory>
#include <stdexcept>
#include <vector>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/simulation/simulation_hd.cuh>

#include "boundary/pml_corrector_hd.cuh"
#include "domain/domain_hd.cuh"
#include "material/ade_method/ade_method_hd.cuh"
#include "material/ade_method/debye_ade_method_hd.cuh"
#include "material/ade_method/drude_ade_method_hd.cuh"
#include "material/ade_method/m_lor_ade_method_hd.cuh"
#include "monitor/movie_monitor_hd.cuh"
#include "nf2ff/frequency_domain/nf2ff_frequency_domain_hd.cuh"
#include "nf2ff/time_domain/nf2ff_time_domain_hd.cuh"
#include "updator/ade_updator/ade_updator_hd.cuh"
#include "updator/ade_updator/debye_ade_updator_hd.cuh"
#include "updator/ade_updator/drude_ade_updator_hd.cuh"
#include "updator/ade_updator/m_lor_ade_updator_hd.cuh"
#include "updator/basic_updator/basic_updator_3d_hd.cuh"
#include "updator/basic_updator/basic_updator_te_hd.cuh"
#include "waveform_source/tfsf/tfsf_corrector_hd.cuh"
#include "xfdtd_cuda/index_task.cuh"

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

  if (_ade_method_storage_hd != nullptr) {
    _ade_method_storage_hd->copyHostToDevice();
  }
}

auto SimulationHD::copyDeviceToHost() -> void {
  if (host() == nullptr) {
    throw std::runtime_error("Host is nullptr");
  }

  if (_grid_space_hd == nullptr) {
    throw std::runtime_error("GridSpaceHD is nullptr");
  }

  if (_calculation_param_hd == nullptr) {
    throw std::runtime_error("CalculationParamHD is nullptr");
  }

  if (_emf_hd == nullptr) {
    throw std::runtime_error("EMFHD is nullptr");
  }

  _grid_space_hd->copyDeviceToHost();
  _calculation_param_hd->copyDeviceToHost();
  _emf_hd->copyDeviceToHost();

  if (_ade_method_storage_hd != nullptr) {
    _ade_method_storage_hd->copyDeviceToHost();
  }
}

auto SimulationHD::releaseDevice() -> void {
  if (_grid_space_hd != nullptr) {
    _grid_space_hd->releaseDevice();
  }

  if (_calculation_param_hd != nullptr) {
    _calculation_param_hd->releaseDevice();
  }

  if (_emf_hd != nullptr) {
    _emf_hd->releaseDevice();
  }

  if (_ade_method_storage_hd != nullptr) {
    _ade_method_storage_hd->releaseDevice();
  }
}

auto SimulationHD::run(Index time_step) -> void {
  for (auto&& v : host()->visitors()) {
    v->initStep(SimulationInitFlag::SimulationStart);
  }
  std::vector<std::unique_ptr<TFSFCorrectorHD>> tfsf_hd;
  init(time_step);
  copyHostToDevice();

  auto task =
      xfdtd::cuda::IndexTask{IndexRange{0, _grid_space_hd->host()->sizeX()},
                             IndexRange{0, _grid_space_hd->host()->sizeY()},
                             IndexRange{0, _grid_space_hd->host()->sizeZ()}};
  // Create domain
  std::unique_ptr<BasicUpdatorHD> basic_updator_hd = nullptr;
  std::unique_ptr<ADEUpdatorHD> ade_updator_hd = nullptr;
  auto&& domain_hd = makeDomainHD(task, ade_updator_hd, basic_updator_hd);

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
  auto nf2ff_td_hd = getNF2FFTD();
  for (auto&& nf2ff : nf2ff_td_hd) {
    nf2ff->copyHostToDevice();
    domain_hd->addNF2FFTimeDoaminAgency(nf2ff->agency());
  }
  for (auto&& v : host()->visitors()) {
    domain_hd->addSimulationFlagVisitor(v);
  }

  domain_hd->run();

  copyDeviceToHost();

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
  for (auto&& n : nf2ff_td_hd) {
    n->copyDeviceToHost();
  }
  for (auto&& v : host()->visitors()) {
    v->initStep(SimulationInitFlag::SimulationEnd);
  }
  {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }
}

auto SimulationHD::init(Index time_step) -> void {
  host()->init(time_step);
  _grid_space_hd = std::make_shared<GridSpaceHD>(host()->gridSpace().get());
  _calculation_param_hd =
      std::make_shared<CalculationParamHD>(host()->calculationParam().get());
  _emf_hd = std::make_shared<EMFHD>(host()->emf().get());

  _ade_method_storage_hd = makeADEMethodStorageHD();
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

auto SimulationHD::getNF2FFTD()
    -> std::vector<std::unique_ptr<NF2FFTimeDomainHD>> {
  auto nf2ffs = host()->nf2ffs();
  if (nf2ffs.size() == 0) {
    return {};
  }

  std::vector<std::unique_ptr<NF2FFTimeDomainHD>> nf2ff_td_hd;
  for (const auto& n : nf2ffs) {
    auto td = dynamic_cast<xfdtd::NFFFTTimeDomain*>(n.get());
    if (td == nullptr) {
      continue;
    }

    nf2ff_td_hd.emplace_back(std::make_unique<NF2FFTimeDomainHD>(
        td, _grid_space_hd, _calculation_param_hd, _emf_hd));
  }
  if (!nf2ff_td_hd.empty()) {
    std::cerr << "Warning: NF2FFTimeDomainHD is not implemented for parallel "
                 "processing\n";
  }

  return nf2ff_td_hd;
}

auto SimulationHD::makeADEMethodStorageHD()
    -> std::unique_ptr<ADEMethodStorageHD> {
  if (host()->aDEMethodStorage() == nullptr) {
    return {};
  }

  // return
  // std::make_unique<ADEMethodStorageHD>(host()->aDEMethodStorage().get());
  auto host_storage = host()->aDEMethodStorage();

  {  // is drude method
    auto host_drude_storage =
        std::dynamic_pointer_cast<xfdtd::DrudeADEMethodStorage>(host_storage);
    if (host_drude_storage != nullptr) {
      return std::make_unique<DrudeADEMethodStorageHD>(
          host_drude_storage.get());
    }
  }

  {
    auto host_debye_storage =
        std::dynamic_pointer_cast<xfdtd::DebyeADEMethodStorage>(host_storage);
    if (host_debye_storage != nullptr) {
      return std::make_unique<DebyeADEMethodStorageHD>(
          host_debye_storage.get());
    }
  }

  {
    auto m_lor_storage =
        std::dynamic_pointer_cast<xfdtd::MLorentzADEMethodStorage>(
            host_storage);

    if (m_lor_storage != nullptr) {
      return std::make_unique<MLorentzADEMethodStorageHD>(m_lor_storage.get());
    }
  }

  throw std::runtime_error(
      "XFDTD CUDA SimulationHD::makeADEMethodStorageHD: "
      "Invalid ADEMethodStorage");
}

auto SimulationHD::makeDomainHD(
    IndexTask task, std::unique_ptr<ADEUpdatorHD>& ade_updator_hd,
    std::unique_ptr<BasicUpdatorHD>& basic_updator_hd)
    -> std::unique_ptr<DomainHD> {
  if (!task.valid()) {
    return nullptr;
  }

  std::unique_ptr<DomainHD> domain_hd = nullptr;

  ade_updator_hd = nullptr;
  basic_updator_hd = nullptr;

  ade_updator_hd = makeADEUpdatorHD(task);

  if (ade_updator_hd == nullptr) {
    basic_updator_hd = makeBasicUpdatorHD(task);
  }

  if (basic_updator_hd != nullptr) {
    basic_updator_hd->copyHostToDevice();
    domain_hd = std::make_unique<DomainHD>(
        _grid_dim, _block_dim, _grid_space_hd, _calculation_param_hd, _emf_hd,
        (basic_updator_hd->getUpdatorAgency()));
  } else if (ade_updator_hd != nullptr) {
    ade_updator_hd->copyHostToDevice();
    domain_hd = std::make_unique<DomainHD>(
        _grid_dim, _block_dim, _grid_space_hd, _calculation_param_hd, _emf_hd,
        (ade_updator_hd->getUpdatorAgency()));
  } else {
    throw std::runtime_error("XFDTD CUDA SimulationHD: Can't find UpdatorHD");
  }

  return domain_hd;
}

auto SimulationHD::makeBasicUpdatorHD(IndexTask task)
    -> std::unique_ptr<BasicUpdatorHD> {
  if (_ade_method_storage_hd != nullptr) {
    return {};
  }

  if (_grid_space_hd->host()->dimension() == xfdtd::GridSpace::Dimension::TWO) {
    return std::make_unique<BasicUpdatorTEHD>(task, _grid_space_hd,
                                              _calculation_param_hd, _emf_hd);
  }

  if (_grid_space_hd->host()->dimension() ==
      xfdtd::GridSpace::Dimension::THREE) {
    return std::make_unique<BasicUpdator3DHD>(task, _grid_space_hd,
                                              _calculation_param_hd, _emf_hd);
  }

  throw std::runtime_error(
      "XFDTD CUDA SimulationHD::makeBasicUpdatorHD: Invalid grid dimension");
}

auto SimulationHD::makeADEUpdatorHD(IndexTask task)
    -> std::unique_ptr<ADEUpdatorHD> {
  if (_ade_method_storage_hd == nullptr) {
    return {};
  }

  {
    auto drude_storage_hd = std::dynamic_pointer_cast<DrudeADEMethodStorageHD>(
        _ade_method_storage_hd);
    if (drude_storage_hd != nullptr) {
      return std::make_unique<DrudeADEUpdatorHD>(task, _grid_space_hd,
                                                 _calculation_param_hd, _emf_hd,
                                                 drude_storage_hd);
    }
  }

  {
    auto debye_storage_hd = std::dynamic_pointer_cast<DebyeADEMethodStorageHD>(
        _ade_method_storage_hd);
    if (debye_storage_hd != nullptr) {
      return std::make_unique<DebyeADEUpdatorHD>(task, _grid_space_hd,
                                                 _calculation_param_hd, _emf_hd,
                                                 debye_storage_hd);
    }
  }

  {
    auto m_lor_storage_hd =
        std::dynamic_pointer_cast<MLorentzADEMethodStorageHD>(
            _ade_method_storage_hd);
    if (m_lor_storage_hd != nullptr) {
      return std::make_unique<MLorentzADEUpdatorHD>(task, _grid_space_hd,
                                                    _calculation_param_hd,
                                                    _emf_hd, m_lor_storage_hd);
    }
  }

  throw std::runtime_error(
      "XFDTD CUDA SimulationHD::makeADEUpdatorHD: Invalid ADEMethodStorage");
}

}  // namespace xfdtd::cuda
