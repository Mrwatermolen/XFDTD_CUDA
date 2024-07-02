#ifndef __XFDTD_CUDA_SIMULATION_HD_CUH__
#define __XFDTD_CUDA_SIMULATION_HD_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/simulation/simulation.h>

#include <memory>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/simulation/simulation.cuh>

#include "xfdtd_cuda/calculation_param/calculation_param_hd.cuh"
#include "xfdtd_cuda/domain/domain_hd.cuh"
#include "xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh"
#include "xfdtd_cuda/grid_space/grid_space_hd.cuh"

namespace xfdtd::cuda {

class SimulationHD : public HostDeviceCarrier<xfdtd::Simulation, void> {
 public:
  SimulationHD(xfdtd::Simulation* host) : HostDeviceCarrier{host} {}

  ~SimulationHD() override { releaseDevice(); }

  auto copyHostToDevice() -> void override {
    if (host() == nullptr) {
      throw std::runtime_error("Host is nullptr");
    }

    _grid_space_hd->copyHostToDevice();
    _calculation_param_hd->copyHostToDevice();
    _emf_hd->copyHostToDevice();
  }

  auto copyDeviceToHost() -> void override {
    if (host() == nullptr) {
      throw std::runtime_error("Host is nullptr");
    }

    _grid_space_hd->copyDeviceToHost();
    _calculation_param_hd->copyDeviceToHost();
    _emf_hd->copyDeviceToHost();
  }

  auto releaseDevice() -> void override {
    _grid_space_hd->releaseDevice();
    _calculation_param_hd->releaseDevice();
    _emf_hd->releaseDevice();
  }

  auto run(Index time_step) -> void {
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
        updator_hd->getUpdatorAgency());

    std::cout << "SimulationHD::run() - domain created \n";

    domain_hd->run();
    std::cout << "SimulationHD::run() - domain run \n";
    copyDeviceToHost();
    std::cout << "SimulationHD::run() End!\n";
  }

  auto init(Index time_step) -> void {
    host()->init(time_step);
    _grid_space_hd = std::make_shared<GridSpaceHD>(host()->gridSpace().get());
    _calculation_param_hd =
        std::make_shared<CalculationParamHD>(host()->calculationParam().get());
    _emf_hd = std::make_shared<EMFHD>(host()->emf().get());
  }

  auto setGridDim(dim3 grid_dim) -> void { _grid_dim = grid_dim; }

  auto setBlockDim(dim3 block_dim) -> void { _block_dim = block_dim; }

 private:
  dim3 _grid_dim{1, 1, 1};
  dim3 _block_dim{1, 1, 1};

  std::shared_ptr<GridSpaceHD> _grid_space_hd{};
  std::shared_ptr<CalculationParamHD> _calculation_param_hd{};
  std::shared_ptr<EMFHD> _emf_hd{};
};
};  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_SIMULATION_HD_CUH__
