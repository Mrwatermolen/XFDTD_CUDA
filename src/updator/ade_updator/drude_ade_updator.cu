// #include <xfdtd/coordinate_system/coordinate_system.h>

// #include <xfdtd_cuda/common.cuh>

// #include "updator/ade_updator/drude_ade_updator.cuh"
// #include "updator/ade_updator/drude_ade_updator_agency.cuh"
// #include "updator/ade_updator/template_ade_update_scheme.cuh"

// namespace xfdtd::cuda {

// XFDTD_CUDA_DEVICE auto DrudeADEUpdator::updateE() -> void {
//   // const auto block_range = blockRange();

//   // const auto& node_task = nodeTask();
//   // const auto nx = node_task.xRange().size();
//   // const auto ny = node_task.yRange().size();
//   // const auto nz = node_task.zRange().size();

//   // TemplateADEUpdateScheme::updateE<DrudeADEUpdator, xfdtd::Axis::XYZ::X, Index>(
//   //     this, block_range.start(), block_range.end(), nx, ny, nz);
//   // TemplateADEUpdateScheme::updateE<DrudeADEUpdator, xfdtd::Axis::XYZ::Y, Index>(
//   //     this, block_range.start(), block_range.end(), nx, ny, nz);
//   // TemplateADEUpdateScheme::updateE<DrudeADEUpdator, xfdtd::Axis::XYZ::Z, Index>(
//   //     this, block_range.start(), block_range.end(), nx, ny, nz);
// }

// XFDTD_CUDA_GLOBAL void kernelCallDrudeADEUpdatorUpdateE(
//     DrudeADEUpdator* updator) {
//   updator->updateE();
// }

// XFDTD_CUDA_GLOBAL void kernelCallDrudeADEUpdatorUpdateH(
//     DrudeADEUpdator* updator) {
//   updator->ADEUpdator::updateH();
// }

// // Agency
// auto DrudeADEUpdatorAgency::updateE(dim3 grid_size, dim3 block_size) -> void {
//   kernelCallDrudeADEUpdatorUpdateE<<<grid_size, block_size>>>(_updator);
// }

// auto DrudeADEUpdatorAgency::updateH(dim3 grid_size, dim3 block_size) -> void {
//   kernelCallDrudeADEUpdatorUpdateH<<<grid_size, block_size>>>(_updator);
// }

// auto DrudeADEUpdatorAgency::setDevice(DrudeADEUpdator* updator) -> void {
//   _updator = updator;
// }

// }  // namespace xfdtd::cuda
