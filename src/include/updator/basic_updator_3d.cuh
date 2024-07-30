#ifndef __XFDTD_CUDA_BASIC_UPDATOR_3D_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_3D_CUH__

#include <xfdtd_cuda/index_task.cuh>

namespace xfdtd::cuda {

class CalculationParam;
class EMF;

class BasicUpdator3D {
  friend class BasicUpdator3DHD;

 public:
  XFDTD_CUDA_DEVICE auto updateH() -> void;

  XFDTD_CUDA_DEVICE auto updateE() -> void;

  XFDTD_CUDA_DUAL auto emf() -> EMF*;

  XFDTD_CUDA_DUAL auto calculationParam() -> CalculationParam*;

 private:
  IndexTask _node_task{};

  CalculationParam* _calculation_param{};
  EMF* _emf{};

  XFDTD_CUDA_DEVICE static auto decomposeRange(IndexRange range, Index id,
                                               Index size) -> IndexRange {
    auto problem_size = range.size();
    auto quotient = problem_size / size;
    auto remainder = problem_size % size;
    auto start = Index{range.start()};
    auto end = Index{range.end()};

    if (id < remainder) {
      start += id * (quotient + 1);
      end = start + quotient + 1;
      return {start, end};
    }

    start += id * quotient + remainder;
    end = start + quotient;
    return {start, end};
  }
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_3D_CUH__
