#ifndef __XFDTD_CUDA_INTERPOLATE_SCHEME_CUH__
#define __XFDTD_CUDA_INTERPOLATE_SCHEME_CUH__

#include <xfdtd/coordinate_system/coordinate_system.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>

namespace xfdtd::interpolate {

// Ey: i, j + 1/2, k. Need i, j+ 1/2, k + 1/2. k
XFDTD_CUDA_DEVICE inline auto interpolateEyFaceX(const auto& ey, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.5 * (ey(i, j, k + 1) + ey(i, j, k));
}

// Ez: i, j, k + 1/2. Need i, j + 1/2, k+ 1/2. j
XFDTD_CUDA_DEVICE inline auto interpolateEzFaceX(const auto& ez, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.5 * (ez(i, j + 1, k) + ez(i, j, k));
}

// Hy: i + 1/2, j, k + 1/2. Need i, j + 1/2, k+1/2. i, j
XFDTD_CUDA_DEVICE inline auto interpolateHyFaceX(const auto& hy, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.25 * (hy(i, j + 1, k) + hy(i, j, k) + hy(i - 1, j + 1, k) +
                 hy(i - 1, j, k));
}

// Hz: i + 1/2, j+ 1/2, k. Need i, j + 1/2, k + 1/2. i, k
XFDTD_CUDA_DEVICE inline auto interpolateHzFaceX(const auto& hz, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.25 * (hz(i, j, k + 1) + hz(i, j, k) + hz(i - 1, j, k + 1) +
                 hz(i - 1, j, k));
}

// Ez: i, j, k + 1/2. Need i + 1/2, j, k + 1/2. i
XFDTD_CUDA_DEVICE inline auto interpolateEzFaceY(const auto& ez, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.5 * (ez(i + 1, j, k) + ez(i, j, k));
}

// Ex: i + 1/2, j, k. Need i + 1/2, j, k + 1/2. k
XFDTD_CUDA_DEVICE inline auto interpolateExFaceY(const auto& ex, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.5 * (ex(i, j, k + 1) + ex(i, j, k));
}

// Hz: i + 1/2, j + 1/2, k. Need i + 1/2, j, k + 1/2. j, k
XFDTD_CUDA_DEVICE inline auto interpolateHzFaceY(const auto& hz, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.25 * (hz(i, j, k + 1) + hz(i, j, k) + hz(i, j - 1, k + 1) +
                 +hz(i, j - 1, k));
}

// Hx: i, j + 1/2, k + 1/2. Need i + 1/2, j, k + 1/2. i, j
XFDTD_CUDA_DEVICE inline auto interpolateHxFaceY(const auto& hx, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.25 * (hx(i + 1, j, k) + hx(i, j, k) + hx(i + 1, j - 1, k) +
                 hx(i, j - 1, k));
}

// Ex: i + 1/2, j, k. Need i + 1/2, j + 1/2, k. j
XFDTD_CUDA_DEVICE inline auto interpolateExFaceZ(const auto& ex, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.5 * (ex(i, j + 1, k) + ex(i, j, k));
}

// Ey: i, j + 1/2, k. Need i + 1/2, j + 1/2, k. i
XFDTD_CUDA_DEVICE inline auto interpolateEyFaceZ(const auto& ey, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.5 * (ey(i + 1, j, k) + ey(i, j, k));
}

// Hx: i, j + 1/2, k + 1/2. Need i + 1/2, j + 1/2, k. i, k
XFDTD_CUDA_DEVICE inline auto interpolateHxFaceZ(const auto& hx, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.25 * (hx(i + 1, j, k) + hx(i, j, k) + hx(i, j, k - 1) +
                 hx(i + 1, j, k - 1));
}

// Hy: i + 1/2, j, k + 1/2. Need i + 1/2, j + 1/2, k. j, k
XFDTD_CUDA_DEVICE inline auto interpolateHyFaceZ(const auto& hy, const auto& i,
                                                 const auto& j, const auto& k) {
  return 0.25 * (hy(i, j, k) + hy(i, j + 1, k) + hy(i, j, k - 1) +
                 hy(i, j + 1, k - 1));
}

template <xfdtd::Axis::XYZ xyz, xfdtd::EMF::Field f>
XFDTD_CUDA_DEVICE inline auto interpolateSurfaceCenter(const auto& data,
                                                       Index i, Index j,
                                                       Index k) {
  if constexpr (xyz == xfdtd::Axis::XYZ::X) {
    if constexpr (f == xfdtd::EMF::Field::HZ) {
      return interpolateHzFaceX(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::HY) {
      return interpolateHyFaceX(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::EZ) {
      return interpolateEzFaceX(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::EY) {
      return interpolateEyFaceX(data, i, j, k);
    } else {
      static_assert(xyz != xfdtd::Axis::XYZ::X, "Invalid field");
    }
  } else if constexpr (xyz == xfdtd::Axis::XYZ::Y) {
    if constexpr (f == xfdtd::EMF::Field::HZ) {
      return interpolateHzFaceY(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::HX) {
      return interpolateHxFaceY(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::EZ) {
      return interpolateEzFaceY(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::EX) {
      return interpolateExFaceY(data, i, j, k);
    } else {
      static_assert(xyz != xfdtd::Axis::XYZ::Y, "Invalid field");
    }
  } else if constexpr (xyz == xfdtd::Axis::XYZ::Z) {
    if constexpr (f == xfdtd::EMF::Field::HX) {
      return interpolateHxFaceZ(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::HY) {
      return interpolateHyFaceZ(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::EX) {
      return interpolateExFaceZ(data, i, j, k);
    } else if constexpr (f == xfdtd::EMF::Field::EY) {
      return interpolateEyFaceZ(data, i, j, k);
    } else {
      static_assert(xyz != xfdtd::Axis::XYZ::Z, "Invalid field");
    }
  } else {
    static_assert(xyz != xfdtd::Axis::XYZ::X && xyz != xfdtd::Axis::XYZ::Y &&
                      xyz != xfdtd::Axis::XYZ::Z,
                  "Invalid field");
  }
}

}  // namespace xfdtd::interpolate

#endif  // __XFDTD_CUDA_INTERPOLATE_SCHEME_CUH__
