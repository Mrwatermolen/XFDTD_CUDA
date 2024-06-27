#ifndef __XFDTD_CUDA_COMMON_CUH__
#define __XFDTD_CUDA_COMMON_CUH__

#include <cstddef>
#include <iostream>
#include <sstream>

namespace xfdtd::cuda {

#define XFDTD_CUDA_GLOBAL __global__
#define XFDTD_CUDA_HOST __host__
#define XFDTD_CUDA_DEVICE __device__
#define XFDTD_CUDA_DUAL __host__ __device__

// define marco for check cuda error
#define XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(call)                                 \
  {                                                                            \
    auto err = call;                                                           \
    if (err != cudaSuccess) {                                                  \
      std::stringstream ss;                                                    \
      ss << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__     \
         << ": " << cudaGetErrorString(err) << "\n";                           \
      std::cerr << ss.str();                                                   \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  }

using SizeType = std::size_t;

// forward declaration
template <typename T, SizeType N> class Tensor;

template <typename T> using Array1D = Tensor<T, 1>;

template <typename T> using Array2D = Tensor<T, 2>;

template <typename T> using Array3D = Tensor<T, 3>;

template <typename T> using Array4D = Tensor<T, 4>;

} // namespace xfdtd::cuda

#endif // __XFDTD_CUDA_COMMON_CUH__
