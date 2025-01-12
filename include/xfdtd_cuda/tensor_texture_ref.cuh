#ifndef __XFDTD_CUDA_TENSOR_TEXTURE_REF_CUH__
#define __XFDTD_CUDA_TENSOR_TEXTURE_REF_CUH__

#include <driver_types.h>
#include <texture_indirect_functions.h>
#include <texture_types.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/fixed_array.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd::cuda {

// only support for float and double
template <typename T, SizeType N>
class TensorTextureRef {
  using DimArray = FixedArray<SizeType, N>;

 public:
  TensorTextureRef() = default;

  XFDTD_CUDA_HOST TensorTextureRef(const TensorHD<T, N> &t_td, DimArray strides)
      : _strides{strides} {
    SizeType size = 1;
    for (SizeType i = 0; i < N; ++i) {
      size *= t_td.shape()[i];
    }
    bind(t_td.deviceData(), size);
  }

  // ~TensorTextureRef() { release(); }

  XFDTD_CUDA_DUAL constexpr static auto dimension() { return N; }

  template <typename... Args>
  XFDTD_CUDA_DUAL auto operator()(Args &&...args) -> T {
    auto offset = dataOffset(_strides.data(), std::forward<Args>(args)...);
#ifdef XFDTD_CORE_SINGLE_PRECISION
    return tex1Dfetch<T>(_texture, offset);
#else
    return fecthForDouble(offset);
#endif
  }

  template <typename... Args>
  XFDTD_CUDA_DUAL auto operator()(Args &&...args) const -> T {
    auto offset = dataOffset(_strides.data(), std::forward<Args>(args)...);
#ifdef XFDTD_CORE_SINGLE_PRECISION
    return tex1Dfetch<T>(_texture, offset);
#else
    return fecthForDouble(offset);
#endif
  }

  XFDTD_CUDA_HOST auto release() -> void {
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaDestroyTextureObject(_texture));
  }

  template <typename Arr>
  XFDTD_CUDA_HOST auto setStrides(const Arr &strides) -> void {
    for (SizeType i = 0; i < N; ++i) {
      _strides[i] = strides[i];
    }
  }

  XFDTD_CUDA_HOST auto setStrides(DimArray strides) -> void {
    _strides = strides;
  }

  XFDTD_CUDA_HOST auto strides() const -> const DimArray & { return _strides; }

  XFDTD_CUDA_HOST auto bind(T *d_data, SizeType size) -> void {
    cudaResourceDesc res_desc{};
    cudaTextureDesc tex_desc{};

    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = d_data;
    res_desc.res.linear.sizeInBytes = size * sizeof(T);

#ifdef XFDTD_CORE_SINGLE_PRECISION
    res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
    res_desc.res.linear.desc.x = 32;
#else
    // use 2D for double precision
    res_desc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    res_desc.res.linear.desc.x = 32;
    res_desc.res.linear.desc.y = 32;
#endif

    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
        cudaCreateTextureObject(&_texture, &res_desc, &tex_desc, nullptr));
  }

  DimArray _strides{};

 private:
  XFDTD_CUDA_DEVICE auto fecthForDouble(SizeType offset) const -> T {
    auto rval = tex1Dfetch<uint2>(_texture, offset);
    auto v = __hiloint2double(rval.y, rval.x);
    return static_cast<T>(v);
  }

  template <SizeType dim>
  XFDTD_CUDA_DUAL static auto rawOffset(const SizeType strides[]) -> SizeType {
    return 0;
  }

  template <SizeType dim, typename Arg, typename... Args>
  XFDTD_CUDA_DUAL static auto rawOffset(const SizeType strides[], Arg &&arg,
                                        Args &&...args) -> SizeType {
    return static_cast<SizeType>(arg) * strides[dim] +
           rawOffset<dim + 1>(strides, std::forward<Args>(args)...);
  }

  template <typename Arg, typename... Args>
  XFDTD_CUDA_DUAL static auto dataOffset(const SizeType stride[], Arg &&arg,
                                         Args &&...args) -> SizeType {
    constexpr SizeType nargs = sizeof...(Args) + 1;
    if constexpr (nargs == dimension()) {
      return rawOffset<static_cast<SizeType>(0)>(stride, std::forward<Arg>(arg),
                                                 std::forward<Args>(args)...);
    }

    static_assert(nargs == dimension(), "Invalid number of arguments");
  }

 private:
  cudaTextureObject_t _texture{};
};

}  // namespace xfdtd::cuda

#endif  //__XFDTD_CUDA_TENSOR_TEXTURE_REF_CUH__