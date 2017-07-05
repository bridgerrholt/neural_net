#ifndef NEURAL_NET_TESTING_ACTIVATION_FUNCTION_H
#define NEURAL_NET_TESTING_ACTIVATION_FUNCTION_H

#include <cmath>

#include "common.h"

namespace brh {
	namespace neural {

FloatType softStep(FloatType value) {
	//std::cout << value << '\n';
	return static_cast<FloatType>(1.0 / (1 + std::pow(MATH_E, -value)));
}



	}
}

#endif
