#ifndef __XFDTD_CUDA_UPDATOR_CUH__
#define __XFDTD_CUDA_UPDATOR_CUH__

#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>
#include <xfdtd_cuda/grid_space/grid_space.cuh>
#include <xfdtd_cuda/index_task.cuh>

namespace xfdtd::cuda {

class Updator {
 public:
  XFDTD_CUDA_HOST Updator(IndexTask task, GridSpace* grid_space,
                          CalculationParam* calculation_param, EMF* emf)
      : _task{task},
        _grid_space{grid_space},
        _calculation_param{calculation_param},
        _emf{emf} {
    if (grid_space == nullptr) {
      throw std::runtime_error("ADEUpdator: grid_space is null");
    }
    if (calculation_param == nullptr) {
      throw std::runtime_error("ADEUpdator: calculation_param is null");
    }
    if (emf == nullptr) {
      throw std::runtime_error("ADEUpdator: emf is null");
    }
  }

  XFDTD_CUDA_DEVICE auto task() const -> IndexTask { return _task; }

  XFDTD_CUDA_DEVICE auto blockRange() const -> IndexRange {
    const auto& task = this->task();
    const auto size =
        task.xRange().size() * task.yRange().size() * task.zRange().size();
    const auto start = Index{0};
    const auto node_range = makeRange(start, size);
    // block
    auto grid_size = (gridDim.x * gridDim.y * gridDim.z);
    auto block_id = blockIdx.x + blockIdx.y * gridDim.x +
                    blockIdx.z * gridDim.x * gridDim.y;
    return decomposeRange(node_range, block_id, grid_size);
  }

  XFDTD_CUDA_DEVICE auto gridSpace() const -> GridSpace* { return _grid_space; }

  XFDTD_CUDA_DEVICE auto calculationParam() const -> CalculationParam* {
    return _calculation_param;
  }

  XFDTD_CUDA_DEVICE auto emf() const -> EMF* { return _emf; }

  XFDTD_CUDA_DEVICE auto gridSpace() { return _grid_space; }

  XFDTD_CUDA_DEVICE auto calculationParam() { return _calculation_param; }

  XFDTD_CUDA_DEVICE auto emf() { return _emf; }

 protected:
  IndexTask _task{};
  GridSpace* _grid_space{};
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

#endif  // __XFDTD_CUDA_UPDATOR_CUH__
