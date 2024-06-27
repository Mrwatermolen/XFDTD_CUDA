#ifndef __XFDTD_CUDA_TENSOR_CUH__
#define __XFDTD_CUDA_TENSOR_CUH__

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/fixed_array.cuh>

namespace xfdtd {

namespace cuda {

template <typename T, SizeType N>
class TensorHD;

template <typename T, SizeType N>
class Tensor {
 public:
  friend class TensorHD<T, N>;

 public:
  using DimArray = FixedArray<SizeType, N>;

  XFDTD_CUDA_DUAL static auto from_shape(DimArray shape) -> Tensor<T, N> {
    return Tensor<T, N>{shape};
  }

 public:
  Tensor() = default;

  XFDTD_CUDA_DUAL Tensor(DimArray shape)
      : _shape{shape},
        _size{makeSize(shape)},
        _strides{makeStride(shape)},
        _data{new T[size()]} {};

  XFDTD_CUDA_DUAL Tensor(const Tensor &other)
      : _shape{other._shape}, _size{other._size}, _strides{other._strides} {
    _data = new T[size()];
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = other._data[i];
    }
  };

  XFDTD_CUDA_DUAL Tensor(Tensor &&other) noexcept
      : _shape{std::move(other._shape)},
        _size{std::move(other._size)},
        _strides{std::move(other._strides)},
        _data{std::move(other._data)} {
    other._data = nullptr;
  };

  XFDTD_CUDA_DUAL ~Tensor() {
    delete[] _data;
    _data = nullptr;
  }

  XFDTD_CUDA_DUAL auto operator()(const Tensor &other) {
    if (this == &other) {
      return *this;
    }

    _shape = other._shape;
    _size = other._size;
    _strides = other._strides;
    _data = new T[size()];
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = other._data[i];
    }

    return *this;
  }

  XFDTD_CUDA_DUAL auto operator()(Tensor &&other) noexcept {
    if (this == &other) {
      return *this;
    }

    _shape = std::move(other._shape);
    _size = std::move(other._size);
    _strides = std::move(other._strides);
    _data = std::move(other._data);

    return *this;
  }

  XFDTD_CUDA_DUAL constexpr static auto dimension() { return N; }

  XFDTD_CUDA_DUAL auto shape() const -> DimArray { return _shape; }

  XFDTD_CUDA_DUAL auto size() const { return _size; }

  XFDTD_CUDA_DUAL auto strides() const -> DimArray { return _strides; }

  XFDTD_CUDA_DUAL auto operator[](SizeType index) -> T & {
    return _data[index];
  }

  XFDTD_CUDA_DUAL auto operator[](SizeType index) const -> const T & {
    return _data[index];
  }

  template <typename... Args>
  XFDTD_CUDA_DUAL auto operator()(Args &&...args) -> T & {
    auto offset = dataOffset(_strides.data(), std::forward<Args>(args)...);
    return _data[offset];
  }

  template <typename... Args>
  XFDTD_CUDA_DUAL auto operator()(Args &&...args) const -> const T & {
    auto offset = dataOffset(_strides.data(), std::forward<Args>(args)...);
    return _data[offset];
  }

  template <typename... Args>
  XFDTD_CUDA_DUAL auto at(Args &&...args) -> T & {
    auto offset = dataOffset(_strides.data(), std::forward<Args>(args)...);
    if (offset >= size()) {
#ifdef __CUDA_ARCH__
      printf("Tensor index out of range");
      asm("trap;");
#else
      throw std::out_of_range("Tensor index out of range");
#endif
    }

    return _data[offset];
  }

  template <typename... Args>
  XFDTD_CUDA_DUAL auto at(Args &&...args) const -> const T & {
    auto offset = dataOffset(_strides.data(), std::forward<Args>(args)...);
    if (offset >= size()) {
#ifdef __CUDA_ARCH__
      printf("Tensor index out of range");
      asm("trap;");
#else
      throw std::out_of_range("Tensor index out of range");
#endif
    }

    return _data[offset];
  }

  XFDTD_CUDA_DUAL auto begin() { return _data; }

  XFDTD_CUDA_DUAL auto end() { return _data + size(); }

  XFDTD_CUDA_DUAL auto begin() const { return _data; }

  XFDTD_CUDA_DUAL auto end() const { return _data + size(); }

  XFDTD_CUDA_DUAL auto cbegin() { return _data; }

  XFDTD_CUDA_DUAL auto cend() { return _data + size(); }

  // XFDTD_CUDA_DUAL auto resize(SizeType size) {}

  XFDTD_CUDA_DUAL auto reshape(DimArray shape) {
#ifdef __CUDA_ARCH__
    if (size() != makeSize(shape)) {
      printf("Cannot reshape tensor to different size");
      asm("trap;");
    }
#else
    if (size() != makeSize(shape)) {
      throw std::invalid_argument("Cannot reshape tensor to different size");
    }
#endif

    _shape = shape;
    _strides = makeStride(shape);
  }

  auto fill(const T &value) {
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = value;
    }
  }

 private:
  DimArray _shape{};
  SizeType _size{};
  DimArray _strides{};

  T *_data{};

  XFDTD_CUDA_DUAL auto makeSize(const DimArray &shape) const -> SizeType {
    if (shape.size() == 0) {
      return 0;
    }

    SizeType size = 1;
    for (const auto &s : shape) {
      size *= s;
    }

    return size;
  }

  XFDTD_CUDA_DUAL auto makeStride(const DimArray &shape) const {
    DimArray stirde{};
    stirde[stirde.size() - 1] = 1;
    for (SizeType i = 1; i < shape.size(); ++i) {
      stirde[stirde.size() - i - 1] =
          stirde[stirde.size() - i] * shape[stirde.size() - i];
    }

    return stirde;
  }

  template <SizeType dim>
  XFDTD_CUDA_DUAL static auto rawOffset(const SizeType strides[])
      -> SizeType {
    return 0;
  }

  template <SizeType dim, typename Arg, typename... Args>
  XFDTD_CUDA_DUAL static auto rawOffset(const SizeType strides[],
                                             Arg &&arg,
                                             Args &&...args) -> SizeType {
    return static_cast<SizeType>(arg) * strides[dim] +
           rawOffset<dim + 1>(strides, std::forward<Args>(args)...);
  }

  template <typename Arg, typename... Args>
  XFDTD_CUDA_DUAL static auto dataOffset(const SizeType stride[],
                                              Arg &&arg,
                                              Args &&...args) -> SizeType {
    constexpr SizeType nargs = sizeof...(Args) + 1;
    if constexpr (nargs == dimension()) {
      return rawOffset<static_cast<SizeType>(0)>(stride, std::forward<Arg>(arg),
                                                 std::forward<Args>(args)...);
    }
  }
};

}  // namespace cuda

}  // namespace xfdtd

#endif  // __XFDTD_CUDA_TENSOR_CUH__
