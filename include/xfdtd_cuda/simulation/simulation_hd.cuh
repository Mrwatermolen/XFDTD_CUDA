#ifndef __XFDTD_CUDA_SIMULATION_HD_CUH__
#define __XFDTD_CUDA_SIMULATION_HD_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>

#include <memory>
#include <vector>
#include <xfdtd_cuda/host_device_carrier.cuh>

namespace xfdtd {

class Simulation;

};  // namespace xfdtd

namespace xfdtd::cuda {

class GridSpaceHD;
class CalculationParamHD;
class EMFHD;
class TFSFCorrectorHD;

template <xfdtd::Axis::XYZ xyz>
class PMLCorrectorHD;

class NF2FFFrequencyDomainHD;
class NF2FFTimeDomainHD;

class SimulationHD : public HostDeviceCarrier<xfdtd::Simulation, void> {
 public:
  SimulationHD(xfdtd::Simulation* host);

  ~SimulationHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto run(Index time_step) -> void;

  auto init(Index time_step) -> void;

  auto setGridDim(dim3 grid_dim) -> void;

  auto setBlockDim(dim3 block_dim) -> void;

 private:
  dim3 _grid_dim{1, 1, 1};
  dim3 _block_dim{1, 1, 1};

  std::shared_ptr<GridSpaceHD> _grid_space_hd{};
  std::shared_ptr<CalculationParamHD> _calculation_param_hd{};
  std::shared_ptr<EMFHD> _emf_hd{};

  auto addTFSFCorrectorHD(
      std::vector<std::unique_ptr<TFSFCorrectorHD>>& tfsf_hd) -> void;

  template <xfdtd::Axis::XYZ xyz>
  auto addPMLBoundaryHD(std::vector<std::unique_ptr<PMLCorrectorHD<xyz>>>&
                            pml_corrector_hd) -> void;

  auto getNF2FFFD()-> std::vector<std::unique_ptr<NF2FFFrequencyDomainHD>>;

  auto getNF2FFTD() -> std::vector<std::unique_ptr<NF2FFTimeDomainHD>>;
};
};  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_SIMULATION_HD_CUH__
