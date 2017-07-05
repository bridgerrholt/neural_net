#ifndef NEURAL_NET_TESTING_SRC_DYNAMIC_NODE_H
#define NEURAL_NET_TESTING_SRC_DYNAMIC_NODE_H

#include <brh/brh_supports.h>

#include "../common.h"

namespace brh {
	namespace neural {
		namespace dynamic {

template <
	template <class> class t_ListInterface = ::ListInterface,
	class t_FloatType = ::FloatType
>
class NodeBase
{
	public:
		struct PointerWeightPair {
			using FloatType = t_FloatType;

			NodeBase * node;
			FloatType  weight;
		};

		using FloatType     = t_FloatType;
		template <class T>
		using ListInterface = t_ListInterface<T>;
		using PointerType   = PointerWeightPair;
		using PointerList   = ListInterface<PointerType>;

		NodeBase() : NodeBase(PointerList()) {}
		NodeBase(PointerList connections) :
			connections_ (std::move(connections)), value_ {0} {
			connections_.reserve(4);
		}

		FloatType   getValue() const { return value_; }
		FloatType & getValue()       { return value_; }

		void setValue  (FloatType value)  { value_ = value; }
		void addToValue(FloatType amount) { value_ += amount; }
		void clearValue() { setValue(0); }

		bool checkReady() {
			return supports::marginCompare(value_, threshold_, threshold_ * 0.01);
		}

		PointerList const & getConnections() const { return connections_; }
		PointerList       & getConnections()       { return connections_; }


	private:
		PointerList connections_;
		FloatType   value_;
		FloatType   threshold_;
};


template <
	class t_ActivationFunc,
	class t_ActivationDerivativeFunc,
	t_ActivationFunc           ACTIVATION_FUNC,
	t_ActivationDerivativeFunc ACTIVATION_DERIVATIVE_FUNC,
  class t_NodeBase = NodeBase<>>
class BasicNode : public t_NodeBase
{
	public:
		using BaseType      = t_NodeBase;
		using FloatType     = typename BaseType::FloatType;
		template <class T>
		using ListInterface = typename BaseType::template ListInterface<T>;
		using PointerType   = typename BaseType::PointerType;
		using PointerList   = typename BaseType::PointerList;

		using ActivationFunc           = t_ActivationFunc;
		using ActivationDerivativeFunc = t_ActivationDerivativeFunc;

		BasicNode() {}
		BasicNode(PointerList connections) :
			BaseType (std::move(connections)) {}


		void applyActivation() {
			BaseType::setValue(ACTIVATION_FUNC(BaseType::getValue()));
		}


};

		}
	}
}

#endif
