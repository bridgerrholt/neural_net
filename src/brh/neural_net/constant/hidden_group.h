#ifndef NEURAL_NET_TESTING_HIDDEN_GROUP_H
#define NEURAL_NET_TESTING_HIDDEN_GROUP_H

#include <iostream>

#include "../common.h"

namespace brh {
	namespace neural {
		namespace constant {

template <
	class t_NodeType,
	template <class T> class t_ListInterface = ::ListInterface
>
class HiddenGroup
{
	public:
		template <class T>
		using ListInterface = t_ListInterface<T>;


		using NodeType  = t_NodeType;
		using NodePtr   = NodeType *;
		using NodeReference   = NodeType &;
		using FloatType = typename NodeType::FloatType;

		using FloatList = ListInterface<FloatType>;

		using FloatPtr      = FloatType       *;
		using ConstFloatPtr = FloatType const *;

		using FloatRef = FloatType &;

		// layerCount may need to be > 1.
		constexpr HiddenGroup(std::size_t inputNodeCount,
		                      std::size_t outputNodeCount,
		                      std::size_t layerCount,
		                      std::size_t nodesPerLayer) :
			inputNodeCount_  {inputNodeCount},
			outputNodeCount_ {outputNodeCount},
			layerCount_      {layerCount},
			nodesPerLayer_   {nodesPerLayer},
			buffer_          (getFloatCount()) { generateBuffer(); }

		std::size_t getInputNodeCount()  const { return inputNodeCount_; }
		std::size_t getOutputNodeCount() const { return outputNodeCount_; }
		std::size_t getLayerCount()      const { return layerCount_; }
		std::size_t getNodesPerLayer()   const { return nodesPerLayer_; }

		FloatList execute(NodePtr nodes, FunctionType activation) {
			auto nonTerm = getNonTerminalElement(0, 0);
			std::size_t layerIndex {0};

			for (std::size_t i {0}; i < getNodesPerLayer(); ++i) {
				auto & node = getNonTerminalElement(layerIndex, i);
				node.clearValue();

				for (std::size_t j {0}; j < getInputNodeCount(); ++j) {
					//std::cout << nodes[j].getValue() << " " << *getInputWeight(j, i) << '\n';
					node.addToValue(nodes[j].getValue() * (*getInputWeight(j, i)));
				}

				applyActivation(node, activation);
			}

			++layerIndex;

			while (layerIndex < getNonTerminalLayerCount()) {
				for (std::size_t i {0}; i < getNodesPerLayer(); ++i) {
					auto & node = getNonTerminalElement(layerIndex, i);
					node.clearValue();

					for (std::size_t j {0}; j < getNodesPerLayer(); ++j) {
						node.addToValue(
							getNonTerminalElement(layerIndex, j).getWeightedValue(i)
						);
					}

					applyActivation(node, activation);
				}

				++layerIndex;
			}

			std::size_t lastNonTerminal {layerIndex - 1};

			for (std::size_t i {0}; i < getNodesPerLayer(); ++i) {
				auto & node = getTerminalElement(i);
				node.clearValue();

				for (std::size_t j {0}; j < getNodesPerLayer(); ++j) {
					node.addToValue(
						getNonTerminalElement(lastNonTerminal, j).getWeightedValue(i)
					);
				}

				applyActivation(node, activation);
			}

			FloatList outValues(getOutputNodeCount());

			for (std::size_t i {0}; i < getOutputNodeCount(); ++i) {
				FloatType & value = outValues[i];
				value = 0;

				for (std::size_t j {0}; j < getNodesPerLayer(); ++j) {
					value += getTerminalElement(j).getWeightedValue(i);
				}

				value = activation(value);
			}

			return outValues;
		}

		void applyActivation(NodeType   & node,
		                     FunctionType activation) {
			auto temp = node.getValue();
			node.applyActivation(activation);

			std::stringstream stream;
			stream << temp << " " << node.getValue();
			printThreadLineMt(stream.str());
		}


		// Input
		std::size_t getFirstInputWeightIndex() const {
			return 0;
		}

		std::size_t getInputWeightIndex(std::size_t inputNodeIndex,
		                                std::size_t weightIndex) const {
			return getFirstInputWeightIndex() +
			       inputNodeIndex * getInputElementSize() + weightIndex;
		}

		FloatPtr getInputWeight(std::size_t inputNodeIndex,
		                        std::size_t weightIndex) {
			return getFloat(getInputWeightIndex(inputNodeIndex, weightIndex));
		}


		// Non-terminal
		std::size_t getFirstNonTerminalElementIndex() const {
			return getInputWeightIndex(getInputNodeCount(), 0);
		}

		std::size_t getNonTerminalElementIndex(std::size_t layerIndex,
		                                       std::size_t nodeIndex) const {
			auto x = getFirstNonTerminalElementIndex();
			auto y = layerIndex * getNonTerminalLayerSize();
			auto z = nodeIndex  * getNonTerminalElementSize();
			return getFirstNonTerminalElementIndex() +
		         layerIndex * getNonTerminalLayerSize() +
				     nodeIndex  * getNonTerminalElementSize();
		}

		NodeReference getNonTerminalElement(std::size_t layerIndex,
		                                   std::size_t nodeIndex) {
			return *getNode(getNonTerminalElementIndex(layerIndex, nodeIndex));
		}


		// Terminal
		std::size_t getFirstTerminalElementIndex() const {
			return getNonTerminalElementIndex(getNonTerminalLayerCount(), 0);
		}

		std::size_t getTerminalElementIndex(std::size_t nodeIndex) const {
			auto x = getFirstTerminalElementIndex();
			return getFirstTerminalElementIndex() +
		         nodeIndex * getTerminalElementSize();
		}

		NodeReference getTerminalElement(std::size_t nodeIndex) {
			return *getNode(getTerminalElementIndex(nodeIndex));
		}


		// Input
		std::size_t getInputElementSize() const {
			return getNodesPerLayer();
		}

		std::size_t getInputLayerSize() const {
			return getInputElementSize() * getInputNodeCount();
		}


		// Non-terminal
		std::size_t getNonTerminalElementSize() const {
			return 1 + getNodesPerLayer();
		}

		std::size_t getNonTerminalLayerSize() const {
			return getNonTerminalElementSize() * getNodesPerLayer();
		}

		std::size_t getNonTerminalGroupSize() const {
			return getNonTerminalLayerSize() * (getNonTerminalLayerCount());
		}

		std::size_t getNonTerminalLayerCount() const {
			return getLayerCount() - 1;
		}


		// Terminal
		std::size_t getTerminalElementSize() const {
			return 1 + getOutputNodeCount();
		}

		std::size_t getTerminalLayerSize() const {
			return getTerminalElementSize() * getNodesPerLayer();
		}


		std::size_t getInputWeightCount() const {
			return getNodesPerLayer() * getInputNodeCount();
		}

		std::size_t getOutputWeightCount() const {
			return getNodesPerLayer() * getOutputNodeCount();
		}

		std::size_t getWeightsPerLayer() const {
			return getNodesPerLayer() * getNodesPerLayer();
		}


	private:
		using BufferType = ListInterface<FloatType>;

		void generateBuffer() {
			BufferType buffer(getFloatCount());

			for (std::size_t i {0}; i < getNonTerminalLayerCount(); ++i) {
				for (std::size_t j {0}; j < getNodesPerLayer(); ++j) {
					*getNode(getNonTerminalElementIndex(i, j)) = {};
				}
			}

			for (std::size_t i {0}; i < getNodesPerLayer(); ++i) {
				*getNode(getTerminalElementIndex(i)) = {};
			}
		}

		FloatPtr getFloat(std::size_t index) {
			return &buffer_.at(index);
		}

		NodePtr getNode(std::size_t index) {
			return reinterpret_cast<NodePtr>(&buffer_.at(index));
		}

		std::size_t getFloatCount() {
			return getInputLayerSize() +
				     getNonTerminalGroupSize() +
				     getTerminalLayerSize();
		}


		std::size_t inputNodeCount_;
		std::size_t outputNodeCount_;
		std::size_t layerCount_;
		std::size_t nodesPerLayer_;

		BufferType buffer_;
};

		}
	}
}

#endif
