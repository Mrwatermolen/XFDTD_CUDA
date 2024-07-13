#ifndef __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_HD_CUH__
#define __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_HD_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>

#include <memory>
#include <vector>

#include "xfdtd_cuda/calculation_param/calculation_param_hd.cuh"
#include "xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh"
#include "xfdtd_cuda/grid_space/grid_space_hd.cuh"
#include "xfdtd_cuda/host_device_carrier.cuh"

namespace xfdtd {

class NFFFTFrequencyDomain;

}  // namespace xfdtd

namespace xfdtd::cuda {

struct SurfaceCurrentSet;
class NF2FFFrequencyDomainAgency;

class NF2FFFrequencyDomainHD
    : public HostDeviceCarrier<NFFFTFrequencyDomain, void> {
  using Host = NFFFTFrequencyDomain;
  using Device = void;

 public:
  NF2FFFrequencyDomainHD(
      Host* host, std::shared_ptr<const GridSpaceHD> grid_space_hd,
      std::shared_ptr<const CalculationParamHD> calculation_param_hd,
      std::shared_ptr<const EMFHD> emf_hd);

  ~NF2FFFrequencyDomainHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto agencies() -> std::vector<NF2FFFrequencyDomainAgency*>&;

 private:
  std::shared_ptr<const GridSpaceHD> _grid_space_hd{};
  std::shared_ptr<const CalculationParamHD> _calculation_param_hd{};
  std::shared_ptr<const EMFHD> _emf_hd{};

  std::vector<std::unique_ptr<SurfaceCurrentSet>> _surface_current_set{};
  std::vector<NF2FFFrequencyDomainAgency*> _agencies{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_HD_CUH__
