#ifndef __XFDTD_CUDA_BASIC_UPDATOR_TE_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_TE_CUH__

#include <xfdtd/common/index_task.h>
#include <xfdtd/common/type_define.h>

#include <tuple>
#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>
#include <xfdtd_cuda/grid_space/grid_space.cuh>
#include <xfdtd_cuda/index_task.cuh>

namespace xfdtd::cuda {

class BasicUpdatorTE {
  friend class BasicUpdatorTEHD;

 public:
  XFDTD_CUDA_DEVICE auto updateH() -> void;

  XFDTD_CUDA_DEVICE auto updateE() -> void;

  XFDTD_CUDA_DEVICE auto task() const -> IndexTask;

  XFDTD_CUDA_DUAL auto emf() { return _emf; }

  XFDTD_CUDA_DUAL auto gridSpace() { return _grid_space; }

  XFDTD_CUDA_DUAL auto calculationParam() { return _calculation_param; }

 private:
  IndexTask _node_task{};
  GridSpace* _grid_space{};
  CalculationParam* _calculation_param{};
  xfdtd::cuda::EMF* _emf{};

  XFDTD_CUDA_DEVICE static auto decomposeTask(IndexTask task, Index id,
                                              Index size_x, Index size_y,
                                              Index size_z) -> IndexTask {
    auto [id_x, id_y, id_z] = columnMajorToRowMajor(id, size_x, size_y, size_z);
    auto x_range = decomposeRange(task.xRange(), id_x, size_x);
    auto y_range = decomposeRange(task.yRange(), id_y, size_y);
    auto z_range = decomposeRange(task.zRange(), id_z, size_z);
    return {x_range, y_range, z_range};
  }

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

  template <typename T>
  XFDTD_CUDA_DEVICE static constexpr auto columnMajorToRowMajor(
      T index, T size_x, T size_y, T size_z) -> std::tuple<T, T, T> {
    return std::make_tuple(index / (size_y * size_z),
                           (index % (size_y * size_z)) / size_z,
                           index % size_z);
  }
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_TE_CUH__
