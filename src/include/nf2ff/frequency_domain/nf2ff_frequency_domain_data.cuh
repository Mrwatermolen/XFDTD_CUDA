#ifndef __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_DATA_CUH__
#define __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_DATA_CUH__

#include <thrust/complex.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/tensor.cuh>

#include "xfdtd_cuda/calculation_param/calculation_param.cuh"
#include "xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh"
#include "xfdtd_cuda/grid_space/grid_space.cuh"
#include "xfdtd_cuda/index_task.cuh"

namespace xfdtd::cuda {

template <xfdtd::Axis::Direction D>
class NF2FFFrequencyDomainData {
  inline constexpr static xfdtd::Axis::XYZ xyz =
      xfdtd::Axis::fromDirectionToXYZ<D>();
  inline constexpr static xfdtd::Axis::XYZ xyz_a =
      xfdtd::Axis::tangentialAAxis<xyz>();
  inline constexpr static xfdtd::Axis::XYZ xyz_b =
      xfdtd::Axis::tangentialBAxis<xyz>();

 public:
  XFDTD_CUDA_DEVICE auto task() const -> IndexTask;

  XFDTD_CUDA_DEVICE auto ds(Index i, Index j, Index k) const -> Real;

  IndexTask _task{};
  const GridSpace* _grid_space{nullptr};
  const CalculationParam* _calculation_param{nullptr};
  const EMF* _emf{nullptr};
  xfdtd::cuda::Array3D<thrust::complex<Real>>*_ja, *_jb, *_ma, *_mb;

  const xfdtd::cuda::Array1D<thrust::complex<Real>>*_transform_e, *_transform_h;

  XFDTD_CUDA_DEVICE auto calculateJ() -> void;

  XFDTD_CUDA_DEVICE auto calculateM() -> void;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_DATA_CUH__
