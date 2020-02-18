#include "optimize/Accumulator.h"

namespace mdso::optimize {

template <>
Accumulator<T>::Accumulator()
    : mWasUsed(false)
    , mAccumulated(0) {}

} // namespace mdso::optimize
