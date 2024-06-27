#ifndef __XFDTD_CUDA_FIXED_ARRAY_CUH__
#define __XFDTD_CUDA_FIXED_ARRAY_CUH__

#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <xfdtd_cuda/common.cuh>

namespace xfdtd {

namespace cuda {

template <typename T, SizeType S>
class FixedArray {
 public:
  FixedArray() = default;

  XFDTD_CUDA_DUAL FixedArray(std::initializer_list<T> list) {
    SizeType i = 0;
    for (auto &item : list) {
      if (size() <= i) {
        break;
      }

      _data[i++] = item;
    }
  }

  XFDTD_CUDA_DUAL FixedArray(const FixedArray &other) {
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = other._data[i];
    }
  }

  XFDTD_CUDA_DUAL FixedArray(FixedArray &&other) noexcept {
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = std::move(other._data[i]);
    }
  }

  XFDTD_CUDA_DUAL auto operator=(FixedArray &&other) noexcept
      -> FixedArray & {
    if (this != &other) {
      for (SizeType i = 0; i < size(); ++i) {
        _data[i] = std::move(other._data[i]);
      }
    }
    return *this;
  }

  XFDTD_CUDA_DUAL auto operator=(const FixedArray &other) -> FixedArray & {
    if (this != &other) {
      for (SizeType i = 0; i < size(); ++i) {
        _data[i] = other._data[i];
      }
    }
    return *this;
  }

  XFDTD_CUDA_DUAL auto operator[](SizeType index) -> T & {
    return _data[index];
  }

  XFDTD_CUDA_DUAL auto operator[](SizeType index) const -> const T & {
    return _data[index];
  }

  XFDTD_CUDA_DUAL auto at(SizeType index) -> T & {
    if (index >= size()) {
#if defined(__CUDA_ARCH__)
      printf("Index out of range\n");
      asm("trap;");
#else
      throw std::out_of_range("Index out of range");
#endif
    }
    return _data[index];
  }

  XFDTD_CUDA_DUAL auto at(SizeType index) const -> const T & {
    if (index >= size()) {
#if defined(__CUDA_ARCH__)
      printf("Index out of range\n");
      asm("trap;");
#else
      throw std::out_of_range("Index out of range");
#endif
    }
    return _data[index];
  }

  XFDTD_CUDA_DUAL static constexpr auto size() -> SizeType { return S; }

  XFDTD_CUDA_DUAL auto begin() -> T * { return _data; }

  XFDTD_CUDA_DUAL auto end() -> T * { return _data + size(); }

  XFDTD_CUDA_DUAL auto begin() const -> const T * { return _data; }

  XFDTD_CUDA_DUAL auto end() const -> const T * { return _data + size(); }

  XFDTD_CUDA_DUAL auto cbegin() const -> const T * { return _data; }

  XFDTD_CUDA_DUAL auto cend() const -> const T * { return _data + size(); }

  XFDTD_CUDA_DUAL auto data() const -> const T * { return _data; }

  XFDTD_CUDA_DUAL auto data() -> T * { return _data; }

 private:
  T _data[S] = {};
};

}  // namespace cuda

}  // namespace xfdtd

#endif  // __XFDTD_CUDA_FIXED_ARRAY_CUH__