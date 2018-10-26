#include "system/SphericalPlus.h"

namespace fishdso {

// clang-format off
Mat33 SphericalPlus::degenerateR =
    (Mat33() << 1.0,  0.0,  0.0,
                0.0, -1.0,  0.0,
                0.0,  0.0, -1.0).finished();
// clang-format on

} // namespace fishdso
