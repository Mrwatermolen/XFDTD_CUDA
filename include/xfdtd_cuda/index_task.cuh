#ifndef __XFDTD_CUDA_INDEX_TASK_CUH__
#define __XFDTD_CUDA_INDEX_TASK_CUH__

#include <xfdtd/common/type_define.h>

#include <sstream>
#include <xfdtd_cuda/common.cuh>

namespace xfdtd {

namespace cuda {

template <typename T>
struct Range {
  XFDTD_CUDA_DUAL auto operator+(T value) const {
    return Range{_start + value, _end + value};
  }

  XFDTD_CUDA_DUAL auto operator-(T value) const {
    return Range{_start - value, _end - value};
  }

  XFDTD_CUDA_DUAL auto start() const { return _start; }

  XFDTD_CUDA_DUAL auto end() const { return _end; }

  XFDTD_CUDA_DUAL auto size() const { return _end - _start; }

  XFDTD_CUDA_DUAL auto valid() const { return _start < _end; }

  XFDTD_CUDA_HOST auto toString() const {
    std::stringstream ss;
    ss << "[" << _start << ", " << _end << ")";
    return ss.str();
  }

  T _start{};
  T _end{};
};

template <typename T>
struct Task {
  Range<T> _x_range{};
  Range<T> _y_range{};
  Range<T> _z_range{};

  XFDTD_CUDA_HOST auto toString() const {
    std::stringstream ss;
    ss << "Task: ";
    ss << "x: " << _x_range.toString() << " ";
    ss << "y: " << _y_range.toString() << " ";
    ss << "z: " << _z_range.toString();
    return ss.str();
  }

  XFDTD_CUDA_DUAL auto xRange() const { return _x_range; }

  XFDTD_CUDA_DUAL auto yRange() const { return _y_range; }

  XFDTD_CUDA_DUAL auto zRange() const { return _z_range; }

  XFDTD_CUDA_DUAL auto valid() const {
    return _x_range.valid() && _y_range.valid() && _z_range.valid();
  }
};

template <typename T>
XFDTD_CUDA_DUAL inline auto makeRange(T start, T end) {
  return Range<T>{start, end};
}

template <typename T>
XFDTD_CUDA_DUAL inline auto makeTask(const Range<T>& x_range,
                                          const Range<T>& y_range,
                                          const Range<T>& z_range) {
  return Task<T>{x_range, y_range, z_range};
}

using IndexRange = Range<Index>;

using IndexTask = Task<Index>;

}  // namespace cuda

}  // namespace xfdtd

#endif  //__XFDTD_CUDA_INDEX_TASK_CUH__
