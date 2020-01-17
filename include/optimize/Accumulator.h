#ifndef INCLUDE_ACCUMULATOR
#define INCLUDE_ACCUMULATOR

template <typename BlockT> class Accumulator {
public:
  Accumulator()
      : mWasUsed(false) {}

  Accumulator &operator+=(const BlockT &block) {
    mAccumulated += block;
    mWasUsed = true;
    return *this;
  }

  bool wasUsed() const { return mWasUsed; }
  const BlockT &accumulated() const { return mAccumulated; }

private:
  BlockT mAccumulated;
  bool mWasUsed;
};

#endif