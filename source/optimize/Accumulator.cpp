#include "optimize/Accumulator.h"

namespace mdso::optimize {

template <> float Accumulator<float>::getZero() { return 0; }
template <> double Accumulator<double>::getZero() { return 0; }

} // namespace mdso::optimize
