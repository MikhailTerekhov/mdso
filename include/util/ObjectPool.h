#include <unordered_map>

namespace fishdso {

template <typename T> class ObjectPool {
public:
  ObjectPool(const T &defaultObject, int initialCapacity)
      : defaultObject(defaultObject) {
    objects.reserve(initialCapacity);
  }

  T &get(int index) {
    auto objectIter = objects.find(index);
    if (objectIter == objects.end())
      objectIter = objects.insert({index, defaultObject}).first;
    return objectIter->second;
  }

private:
  T defaultObject;
  std::unordered_map<int, T> objects;
};

} // namespace fishdso
