#ifndef NEURAL_NET_TESTING_SRC_DYNAMIC_NETWORK_H
#define NEURAL_NET_TESTING_SRC_DYNAMIC_NETWORK_H

#include <limits>

#include "node.h"

namespace brh {
	namespace neural {
		namespace dynamic {

template <
	class t_NodeType,
  class t_Allocator,
  std::size_t t_MAX_NEURON_COUNT = std::numeric_limits<std::size_t>::max(),
  template <class T> class t_ListInterface = ::ListInterface
>
class Network
{
	public:
		using NodeType      = t_NodeType;
		using Allocator     = t_Allocator;
		using NodeBlock     = typename Allocator::BlockType<NodeType>;
		using NodeReference = NodeType &;

		template <class T>
		using ListInterface = t_ListInterface<T>;
		using NodeList      = ListInterface<NodeBlock>;

		template <class ... ArgPack>
		Network(ArgPack ... allocatorArgs) :
			allocator_ (std::forward(allocatorArgs)...) {}

		Network() {}

		~Network() {
			destructNodeList(inputNodes_);
			destructNodeList(outputNodes_);
			destructNodeList(hiddenNodes_);
		}


		NodeReference createInputNode() {

		}

		NodeReference createOutputNode() {

		}

		NodeReference createHiddenNode() {

		}


	private:
		static void pushUniqueToList(NodeBlock block, NodeList & list) {
			auto const listSize = list.size();

			bool found {false};
			for (std::size_t i {0}; i < listSize; ++i) {
				if (list[i] == block.getPtr())
					found = true;
			}

			if (!found)
				list.push_back(block);
		}

		void destructNodeList(NodeList const & list) {
			for (auto i : list)
				allocator_.destruct(i);
		}

		Allocator allocator_;

		NodeList inputNodes_;
		NodeList outputNodes_;
		NodeList hiddenNodes_;
};

		}
	}
}

#endif
