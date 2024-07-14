#ifndef __XFDTD_CUDA_NF2FF_TIME_DOMAIN_AGENCY__CUH__
#define __XFDTD_CUDA_NF2FF_TIME_DOMAIN_AGENCY__CUH__

#include <xfdtd/coordinate_system/coordinate_system.h>

namespace xfdtd::cuda {

template <xfdtd::Axis::Direction D>
class NF2FFTimeDomainData;

class NF2FFTimeDoaminAgency {
 public:
  NF2FFTimeDoaminAgency(
      NF2FFTimeDomainData<xfdtd::Axis::Direction::XN>* data_xn,
      NF2FFTimeDomainData<xfdtd::Axis::Direction::XP>* data_xp,
      NF2FFTimeDomainData<xfdtd::Axis::Direction::YN>* data_yn,
      NF2FFTimeDomainData<xfdtd::Axis::Direction::YP>* data_yp,
      NF2FFTimeDomainData<xfdtd::Axis::Direction::ZN>* data_zn,
      NF2FFTimeDomainData<xfdtd::Axis::Direction::ZP>* data_zp);

  auto update(dim3 grid_dim, dim3 block_dim) -> void;

 private:
  NF2FFTimeDomainData<xfdtd::Axis::Direction::XN>* _xn{nullptr};
  NF2FFTimeDomainData<xfdtd::Axis::Direction::XP>* _xp{nullptr};
  NF2FFTimeDomainData<xfdtd::Axis::Direction::YN>* _yn{nullptr};
  NF2FFTimeDomainData<xfdtd::Axis::Direction::YP>* _yp{nullptr};
  NF2FFTimeDomainData<xfdtd::Axis::Direction::ZN>* _zn{nullptr};
  NF2FFTimeDomainData<xfdtd::Axis::Direction::ZP>* _zp{nullptr};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_NF2FF_TIME_DOMAIN_AGENCY__CUH__
