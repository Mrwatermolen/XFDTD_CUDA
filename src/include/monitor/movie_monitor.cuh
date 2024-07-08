#ifndef __XFDTD_CUDA_MOVIE_MONITOR_CUH__
#define __XFDTD_CUDA_MOVIE_MONITOR_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>

#include <cstdio>
#include <string_view>

#include "xfdtd_cuda/common.cuh"
#include "xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh"
#include "xfdtd_cuda/index_task.cuh"

namespace xfdtd::cuda {

template <typename xfdtd::EMF::Field F>
class MovieMonitor {
  // template <xfdtd::EMF::Field FIELD>
  // friend class MovieMointorHD;

 public:
  MovieMonitor() = default;

  XFDTD_CUDA_DEVICE auto update() -> void;

  XFDTD_CUDA_DEVICE auto task() const -> IndexTask;

  XFDTD_CUDA_HOST auto output(std::string_view path_dir) const -> void;

  XFDTD_CUDA_DEVICE auto nextCount() -> void { ++_frame_count; }

  XFDTD_CUDA_DUAL auto data() { return _data; }

  XFDTD_CUDA_DUAL auto data() const { return _data; }

  XFDTD_CUDA_HOST auto formatFrameCount(Index frame_count) const -> std::string;

  IndexTask _task{};
  const xfdtd::cuda::EMF *_emf{};
  Index _frame_interval{};
  Index _frame_count{};
  xfdtd::cuda::Array4D<Real> *_data{};

 private:
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

#endif  // __XFDTD_CUDA_MOVIE_MONITOR_CUH__
