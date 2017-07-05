#ifndef NEURAL_NET_TESTING_COMMON_H
#define NEURAL_NET_TESTING_COMMON_H

#include <iostream>
#include <sstream>
#include <vector>
#include <functional>
#include <thread>

using FloatType = float;

template <class T>
using ListInterface = std::vector<T>;
using ListType  = ListInterface<FloatType>;


using FunctionType = std::function<FloatType(FloatType)>;

constexpr double MATH_E { 2.7182818284590452353602874 };

template <class T>
void printMt(T const & value) {
	static std::mutex mutex;
	std::lock_guard<std::mutex> guard (mutex);

	std::cout << value << std::flush;
}

template <class T>
void printLineMt(T const & value) {
	std::stringstream stream;
	stream << value;
	printMt(stream.str());
}

template <class T>
void printThreadLineMt(T const & value) {
	std::stringstream stream;
	stream << std::this_thread::get_id() << ": " << value << '\n';
	printMt(stream.str());
}

#endif
