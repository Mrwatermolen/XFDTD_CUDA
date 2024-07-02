#ifndef __XFDTD_CUDA_BASIC_UPDATOR_TE_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_TE_CUH__

#include <xfdtd/common/type_define.h>

#include <tuple>

#include "xfdtd_cuda/calculation_param/calculation_param.cuh"
#include "xfdtd_cuda/common.cuh"
#include "xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh"
#include "xfdtd_cuda/grid_space/grid_space.cuh"
#include "xfdtd_cuda/index_task.cuh"
#include "xfdtd_cuda/updator/update_scheme.cuh"

namespace xfdtd::cuda {

class BasicUpdatorTE {
  friend class BasicUpdatorTEHD;

 public:
  XFDTD_CUDA_DEVICE auto updateH() -> void {
    const auto task = this->task();
    const auto x_range = task.xRange();
    const auto y_range = task.yRange();
    const auto z_range = task.zRange();

    const auto is = x_range.start();
    const auto ie = x_range.end();
    const auto js = y_range.start();
    const auto je = y_range.end();
    const auto ks = z_range.start();
    const auto ke = z_range.end();

    const auto step_i = stepI();
    const auto step_j = stepJ();
    const auto step_k = stepK();

    static bool is_print = false;
    if (!is_print) {
      auto block_id = blockIdx.x + blockIdx.y * gridDim.x +
                      blockIdx.z * gridDim.x * gridDim.y;
      auto thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                       threadIdx.z * blockDim.x * blockDim.y;
      printf(
          "blockIdx: (%d, %d, %d), threadIdx: (%d, %d, %d), block_id: %d, "
          "thread_id: %d, task:[%lu, %lu), [%lu, %lu), [%lu, %lu), step: "
          "[%lu, %lu, %lu]\n",
          blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
          threadIdx.z, block_id, thread_id, is, ie, js, je, ks, ke, step_i,
          step_j, step_k);

      is_print = true;
    }

    update<xfdtd::EMF::Attribute::H, Axis::XYZ::X>(
        *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke,
        step_i, step_j, step_k);
    update<xfdtd::EMF::Attribute::H, Axis::XYZ::Y>(
        *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke,
        step_i, step_j, step_k);
    update<xfdtd::EMF::Attribute::H, Axis::XYZ::Z>(
        *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke,
        step_i, step_j, step_k);
  }

  XFDTD_CUDA_DEVICE auto updateE() -> void {
    const auto task = this->task();
    const auto x_range = task.xRange();
    const auto y_range = task.yRange();
    const auto z_range = task.zRange();

    const auto step_i = stepI();
    const auto step_j = stepJ();
    const auto step_k = stepK();

    const auto is = x_range.start() == 0 ? step_i : x_range.start();
    const auto ie = x_range.end();
    const auto js = y_range.start() == 0 ? step_j : y_range.start();
    const auto je = y_range.end();
    const auto ks = z_range.start();
    const auto ke = z_range.end();

    update<xfdtd::EMF::Attribute::E, Axis::XYZ::Z>(
        *_emf, *_calculation_param->fdtdCoefficient(), is, ie, js, je, ks, ke,
        step_i, step_j, step_k);
  }

  XFDTD_CUDA_DEVICE auto task() const -> IndexTask {
    const auto& node_task = _node_task;
    // blcok
    auto size_x = static_cast<Index>(gridDim.x);
    auto size_y = static_cast<Index>(gridDim.y);
    auto size_z = static_cast<Index>(gridDim.z);
    auto id = blockIdx.x + blockIdx.y * gridDim.x +
              blockIdx.z * gridDim.x * gridDim.y;
    auto block_task = decomposeTask(node_task, id, size_x, size_y, size_z);
    // thread
    size_x = static_cast<Index>(blockDim.x);
    size_y = static_cast<Index>(blockDim.y);
    size_z = static_cast<Index>(blockDim.z);
    id = threadIdx.x + threadIdx.y * blockDim.x +
         threadIdx.z * blockDim.x * blockDim.y;
    auto thread_task = decomposeTask(block_task, id, size_x, size_y, size_z);
    return thread_task;
  }

  XFDTD_CUDA_DUAL auto emf() { return _emf; }

  XFDTD_CUDA_DUAL auto gridSpace() { return _grid_space; }

  XFDTD_CUDA_DUAL auto calculationParam() { return _calculation_param; }

  XFDTD_CUDA_DEVICE auto stepI() const -> Index {
    return static_cast<Index>(blockDim.x);
  }

  XFDTD_CUDA_DEVICE auto stepJ() const -> Index {
    return static_cast<Index>(blockDim.y);
  }

  XFDTD_CUDA_DEVICE auto stepK() const -> Index {
    return static_cast<Index>(blockDim.z);
  }

 private:
  IndexTask _node_task{};
  GridSpaceData* _grid_space{};
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
