#ifndef NEURAL_NET_TESTING_SRC_CONSTANT_NODE_H
#define NEURAL_NET_TESTING_SRC_CONSTANT_NODE_H

#include <limits>

#include <cassert>

#include "../common.h"

namespace brh {
	namespace neural {
		namespace constant {

template <
	class t_FloatType = ::FloatType
>
class BasicNode
{
	public:
		using FloatType     = t_FloatType;

		using WeightPtr      = FloatType       *;
		using ConstWeightPtr = FloatType const *;


		BasicNode() : value_ {0} {}

		template <class FloatList>
		BasicNode(FloatList const & weights) : value_ {0} {
			auto size = weights.size();
			for (std::size_t i {0}; i < size; ++i) {
				*getWeight(i) = weights[i];
			}
		}

		FloatType   getValue() const { return value_; }
		FloatType & getValue()       { return value_; }

		FloatType getWeightedValue(std::size_t weightIndex) const {
			return getValue() * (*getWeight(weightIndex));
		}

		void setValue  (FloatType value)  { value_ = value; }
		void addToValue(FloatType amount) { value_ += amount; }
		void clearValue() { setValue(0); }

		void applyActivation(FunctionType activator) {
			setValue(activator(getValue()));
		}

		WeightPtr      getWeight(std::size_t index) {
			void * nextVoid = static_cast<void *>(this + 1);
			WeightPtr next = static_cast<WeightPtr>(nextVoid);
			return next + index;
		}

		ConstWeightPtr getWeight(std::size_t index) const {
			void const * nextVoid = static_cast<void const *>(this + 1);
			ConstWeightPtr next = static_cast<ConstWeightPtr>(nextVoid);
			return next + index;
		}


	private:
		FloatType value_;
};


using Node = BasicNode<>;


		}
	}
}

#endif