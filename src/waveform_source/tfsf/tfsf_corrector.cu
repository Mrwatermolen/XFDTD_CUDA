#include "waveform_source/tfsf/tfsf_corrector.cuh"
#include "waveform_source/tfsf/tfsf_corrector_agency.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_DEVICE auto TFSFCorrector::globalStartI() const -> Index {
  return _total_task.xRange().start();
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::globalStartJ() const -> Index {
  return _total_task.yRange().start();
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::globalStartK() const -> Index {
  return _total_task.zRange().start();
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::exInc(Index t, Index i, Index j,
                                            Index k) const -> Real {
  i = i - globalStartI() + 1;
  j = j - globalStartJ();
  k = k - globalStartK();
  const auto& projection_x_half = *_projection_x_half;
  const auto& projection_y_int = *_projection_y_int;
  const auto& projection_z_int = *_projection_z_int;

  auto projection{projection_x_half(i) + projection_y_int(j) +
                  projection_z_int(k)};
  auto index{static_cast<Index>(projection)};
  auto weight{projection - index};
  const auto& e_inc = *_e_inc;

  return _transform_e_x *
         ((1 - weight) * e_inc(t, index) + weight * e_inc(t, index + 1));
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::eyInc(Index t, Index i, Index j,
                                            Index k) const -> Real {
  i = i - globalStartI();
  j = j - globalStartJ() + 1;
  k = k - globalStartK();
  const auto& projection_x_int = *_projection_x_int;
  const auto& projection_y_half = *_projection_y_half;
  const auto& projection_z_int = *_projection_z_int;

  auto projection{projection_x_int(i) + projection_y_half(j) +
                  projection_z_int(k)};
  auto index{static_cast<std::size_t>(projection)};
  auto weight{projection - index};
  const auto& e_inc = *_e_inc;

  return _transform_e_y *
         ((1 - weight) * e_inc(t, index) + weight * e_inc(t, index + 1));
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::ezInc(Index t, Index i, Index j,
                                            Index k) const -> Real {
  i = i - globalStartI();
  j = j - globalStartJ();
  k = k - globalStartK() + 1;
  const auto& projection_x_int = *_projection_x_int;
  const auto& projection_y_int = *_projection_y_int;
  const auto& projection_z_half = *_projection_z_half;

  auto projection{projection_x_int(i) + projection_y_int(j) +
                  projection_z_half(k)};
  auto index{static_cast<std::size_t>(projection)};
  auto weight{projection - index};
  const auto& e_inc = *_e_inc;

  return _transform_e_z *
         ((1 - weight) * e_inc.at(t, index) + weight * e_inc.at(t, index + 1));
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::hxInc(Index t, Index i, Index j,
                                            Index k) const -> Real {
  i = i - globalStartI();
  j = j - globalStartJ() + 1;
  k = k - globalStartK() + 1;
  const auto& projection_x_int = *_projection_x_int;
  const auto& projection_y_half = *_projection_y_half;
  const auto& projection_z_half = *_projection_z_half;

  auto projection{projection_x_int(i) + projection_y_half(j) +
                  projection_z_half(k) - 0.5};
  auto index{static_cast<std::size_t>(projection)};
  auto weight{projection - index};
  const auto& h_inc = *_h_inc;

  return _transform_h_x *
         ((1 - weight) * h_inc(t, index) + weight * h_inc(t, index + 1));
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::hyInc(Index t, Index i, Index j,
                                            Index k) const -> Real {
  i = i - globalStartI() + 1;
  j = j - globalStartJ();
  k = k - globalStartK() + 1;
  const auto& projection_x_half = *_projection_x_half;
  const auto& projection_y_int = *_projection_y_int;
  const auto& projection_z_half = *_projection_z_half;

  auto projection{projection_x_half(i) + projection_y_int(j) +
                  projection_z_half(k) - 0.5};
  auto index{static_cast<std::size_t>(projection)};
  auto weight{projection - index};
  const auto& h_inc = *_h_inc;

  return _transform_h_y *
         ((1 - weight) * h_inc(t, index) + weight * h_inc(t, index + 1));
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::hzInc(Index t, Index i, Index j,
                                            Index k) const -> Real {
  i = i - globalStartI() + 1;
  j = j - globalStartJ() + 1;
  k = k - globalStartK();
  const auto& projection_x_half = *_projection_x_half;
  const auto& projection_y_half = *_projection_y_half;
  const auto& projection_z_int = *_projection_z_int;

  auto projection{projection_x_half(i) + projection_y_half(j) +
                  projection_z_int(k) - 0.5};
  auto index{static_cast<std::size_t>(projection)};
  auto weight{projection - index};
  const auto& h_inc = *_h_inc;

  return _transform_h_z *
         ((1 - weight) * h_inc(t, index) + weight * h_inc(t, index + 1));
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::calculationParam() const
    -> const CalculationParam* {
  return _calculation_param;
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::emf() const -> const EMF* { return _emf; }

XFDTD_CUDA_DEVICE auto TFSFCorrector::calculationParam()
    -> const CalculationParam* {
  return _calculation_param;
}

XFDTD_CUDA_DEVICE auto TFSFCorrector::emf() -> EMF* { return _emf; }

// Agency

template <xfdtd::Axis::Direction Direction, xfdtd::EMF::Attribute Attribute,
          xfdtd::Axis::XYZ Axis>
XFDTD_CUDA_GLOBAL auto __tFSFcorrect(TFSFCorrector* device) -> void {
  device->correctTFSF<Direction, Attribute, Axis>(device->task<Direction>(), 0,
                                                  0, 0);
}

auto TFSFCorrector2DAgency::correctE(dim3 grid_size, dim3 block_size) -> void {
  auto grid_dim_x = dim3{1, grid_size.y, grid_size.z};
  auto grid_dim_y = dim3{grid_size.x, 1, grid_size.z};
  auto grid_dim_z = dim3{grid_size.x, grid_size.y, 1};
  auto block_dim_x = dim3{1, block_size.y, block_size.z};
  auto block_dim_y = dim3{block_size.x, 1, block_size.z};
  auto block_dim_z = dim3{block_size.x, block_size.y, 1};

  __tFSFcorrect<Axis::Direction::XN, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Z><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::XP, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Z><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::YN, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Z><<<grid_dim_y, block_dim_y>>>(device());
  __tFSFcorrect<Axis::Direction::YP, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Z><<<grid_dim_y, block_dim_y>>>(device());
}

auto TFSFCorrector2DAgency::correctH(dim3 grid_size, dim3 block_size) -> void {
  auto grid_dim_x = dim3{1, grid_size.y, grid_size.z};
  auto grid_dim_y = dim3{grid_size.x, 1, grid_size.z};
  auto grid_dim_z = dim3{grid_size.x, grid_size.y, 1};
  auto block_dim_x = dim3{1, block_size.y, block_size.z};
  auto block_dim_y = dim3{block_size.x, 1, block_size.z};
  auto block_dim_z = dim3{block_size.x, block_size.y, 1};

  __tFSFcorrect<Axis::Direction::XN, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Y><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::XP, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Y><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::YN, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::X><<<grid_dim_y, block_dim_y>>>(device());
  __tFSFcorrect<Axis::Direction::YP, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::X><<<grid_dim_y, block_dim_y>>>(device());
}

auto TFSFCorrectorAgency::device() -> TFSFCorrector* { return _device; }

auto TFSFCorrectorAgency::device() const -> const TFSFCorrector* {
  return _device;
}

auto TFSFCorrectorAgency::setDevice(TFSFCorrector* device) -> void {
  _device = device;
}

auto TFSFCorrector3DAgency::correctE(dim3 grid_size, dim3 block_size) -> void {
  auto grid_dim_x = dim3{1, grid_size.y, grid_size.z};
  auto grid_dim_y = dim3{grid_size.x, 1, grid_size.z};
  auto grid_dim_z = dim3{grid_size.x, grid_size.y, 1};
  auto block_dim_x = dim3{1, block_size.y, block_size.z};
  auto block_dim_y = dim3{block_size.x, 1, block_size.z};
  auto block_dim_z = dim3{block_size.x, block_size.y, 1};

  __tFSFcorrect<Axis::Direction::XN, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Z><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::XP, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Z><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::YN, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Z><<<grid_dim_y, block_dim_y>>>(device());
  __tFSFcorrect<Axis::Direction::YP, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Z><<<grid_dim_y, block_dim_y>>>(device());

  __tFSFcorrect<Axis::Direction::ZN, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::X><<<grid_dim_z, block_dim_z>>>(device());
  __tFSFcorrect<Axis::Direction::ZP, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::X><<<grid_dim_z, block_dim_z>>>(device());
  __tFSFcorrect<Axis::Direction::YN, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::X><<<grid_dim_y, block_dim_y>>>(device());
  __tFSFcorrect<Axis::Direction::YP, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::X><<<grid_dim_y, block_dim_y>>>(device());

  __tFSFcorrect<Axis::Direction::ZN, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Y><<<grid_dim_z, block_dim_z>>>(device());
  __tFSFcorrect<Axis::Direction::ZP, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Y><<<grid_dim_z, block_dim_z>>>(device());
  __tFSFcorrect<Axis::Direction::XN, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Y><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::XP, xfdtd::EMF::Attribute::E,
                xfdtd::Axis::XYZ::Y><<<grid_dim_x, block_dim_x>>>(device());
}

auto TFSFCorrector3DAgency::correctH(dim3 grid_size, dim3 block_size) -> void {
  auto grid_dim_x = dim3{1, grid_size.y, grid_size.z};
  auto grid_dim_y = dim3{grid_size.x, 1, grid_size.z};
  auto grid_dim_z = dim3{grid_size.x, grid_size.y, 1};
  auto block_dim_x = dim3{1, block_size.y, block_size.z};
  auto block_dim_y = dim3{block_size.x, 1, block_size.z};
  auto block_dim_z = dim3{block_size.x, block_size.y, 1};

  __tFSFcorrect<Axis::Direction::XN, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Z><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::XP, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Z><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::YN, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Z><<<grid_dim_y, block_dim_y>>>(device());
  __tFSFcorrect<Axis::Direction::YP, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Z><<<grid_dim_y, block_dim_y>>>(device());

  __tFSFcorrect<Axis::Direction::ZN, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::X><<<grid_dim_z, block_dim_z>>>(device());
  __tFSFcorrect<Axis::Direction::ZP, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::X><<<grid_dim_z, block_dim_z>>>(device());
  __tFSFcorrect<Axis::Direction::YN, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::X><<<grid_dim_y, block_dim_y>>>(device());
  __tFSFcorrect<Axis::Direction::YP, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::X><<<grid_dim_y, block_dim_y>>>(device());

  __tFSFcorrect<Axis::Direction::ZN, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Y><<<grid_dim_z, block_dim_z>>>(device());
  __tFSFcorrect<Axis::Direction::ZP, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Y><<<grid_dim_z, block_dim_z>>>(device());
  __tFSFcorrect<Axis::Direction::XN, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Y><<<grid_dim_x, block_dim_x>>>(device());
  __tFSFcorrect<Axis::Direction::XP, xfdtd::EMF::Attribute::H,
                xfdtd::Axis::XYZ::Y><<<grid_dim_x, block_dim_x>>>(device());
}

}  // namespace xfdtd::cuda
