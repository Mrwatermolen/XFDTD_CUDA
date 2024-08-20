#ifndef __XFDTD_CUDA_ADE_UPDATOR_CUH__
#define __XFDTD_CUDA_ADE_UPDATOR_CUH__

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/index_task.cuh>

namespace xfdtd::cuda {

class CalculationParam;
class EMF;
class ADEMethodStorage;

class ADEUpdator {
  friend class ADEUpdatorHD;

 public:
  // TODO(franzero): move to father class
  XFDTD_CUDA_DEVICE auto nodeTask() const -> IndexTask { return _node_task; }

  // TODO(franzero): move to father class
  XFDTD_CUDA_DEVICE auto blockRange() const -> IndexRange;

  // TODO(franzero): move to father class
 XFDTD_CUDA_DEVICE auto emf() { return _emf; }

  // TODO(franzero): move to father class
  XFDTD_CUDA_DEVICE auto calculationParam() { return _calculation_param; }

  // TODO(franzero): move to father class
  XFDTD_CUDA_DEVICE auto storage() { return _ade_method_storage; }

  // TODO(franzero): move to father class
  XFDTD_CUDA_DEVICE auto updateH() -> void;

 protected:
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

  IndexTask _node_task{};

  CalculationParam* _calculation_param{};
  EMF* _emf{};
  ADEMethodStorage* _ade_method_storage{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_ADE_UPDATOR_CUH__
