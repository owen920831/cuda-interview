#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

inline std::vector<int> lab_versions(int argc, char** argv, int count) {
  if (argc == 1) {
    std::vector<int> versions(count);
    for (int i = 0; i < count; ++i) versions[i] = i;
    return versions;
  }
  if (argc != 3 || std::string(argv[1]) != "--version") {
    std::cerr << "usage: " << argv[0] << " [--version all|0|1|2]\n";
    std::exit(EXIT_FAILURE);
  }
  const std::string value = argv[2];
  if (value == "all") {
    std::vector<int> versions(count);
    for (int i = 0; i < count; ++i) versions[i] = i;
    return versions;
  }
  const int version = std::stoi(value[0] == 'v' ? value.substr(1) : value);
  if (version < 0 || version >= count) {
    std::cerr << "version out of range\n";
    std::exit(EXIT_FAILURE);
  }
  return {version};
}
