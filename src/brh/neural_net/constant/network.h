#ifndef NEURAL_NET_TESTING_SRC_CONSTANT_NETWORK_H
#define NEURAL_NET_TESTING_SRC_CONSTANT_NETWORK_H

#include <limits>
#include <cassert>
#include <thread>
#include <future>

#include <brh/supports/round_up_to_multiple.h>

#include "../common.h"

#include "hidden_group.h"

namespace brh {
	namespace neural {
		namespace constant {

template <
	class t_NodeType,
	template <class T> class t_ListInterface = ::ListInterface
>
class Network
{
	public:
		template <class T>
		using ListInterface = t_ListInterface<T>;


		using NodeType      = t_NodeType;
		using NodeReference = NodeType &;

		using FloatType = typename NodeType::FloatType;
		using FloatList = ListInterface<FloatType>;

		using NodeList        = ListInterface<NodeType>;
		using HiddenGroupType = HiddenGroup<NodeType, t_ListInterface>;


		Network(std::size_t hiddenGroupCount,
		        std::size_t inputNodeCount,
		        std::size_t outputNodeCount,
		        std::size_t hiddenLayerCount,
		        std::size_t nodesPerHiddenLayer) :
			inputNodes_      (inputNodeCount),
			outputNodes_     (outputNodeCount),
			hiddenGroupList_ (hiddenGroupCount, {
				inputNodeCount, outputNodeCount,
				hiddenLayerCount, nodesPerHiddenLayer }),
			futureList_      (hiddenGroupCount) {}


		void execute(FunctionType activation) {
			auto size = getHiddenGroupCount();

			for (std::size_t i {0}; i < size; ++i) {
				futureList_[i] = std::async(
					std::launch::async,
					&HiddenGroupType::execute, &hiddenGroupList_[i],
					inputNodes_.data(), activation
				);
			}

			std::vector<FloatList> outputValues (size);

			for (std::size_t i {0}; i < size; ++i) {
				outputValues[i] = futureList_[i].get();
			}

			for (std::size_t i {0}; i < getOutputNodeCount(); ++i) {
				auto & node = getOutputNode(i);

				node.clearValue();

				for (std::size_t j {0}; j < size; ++j) {
					node.addToValue(outputValues[j][i]);
				}

				node.applyActivation(activation);
			}
		}

		// If too small, only the first n items are affected,
		// if too large the list is simply cut off.
		void setInputNodes(NodeList nodes) {
			auto newSize = nodes.size();
			auto oldSize = inputNodes_.size();

			std::size_t i {0};

			while (i < newSize && i < oldSize) {
				inputNodes_[i] = nodes[i];
				++i;
			}
		}


		NodeReference getInputNode(std::size_t index) {
			return getItem(inputNodes_, index);
		}

		NodeReference getOutputNode(std::size_t index) {
			return getItem(outputNodes_, index);
		}

		HiddenGroupType & getHiddenGroup(std::size_t index) {
			return getItem(hiddenGroupList_, index);
		}


		std::size_t getInputNodeCount() const {
			return inputNodes_.size();
		}

		std::size_t getOutputNodeCount() const {
			return outputNodes_.size();
		}

		std::size_t getHiddenGroupCount() const {
			return hiddenGroupList_.size();
		}


	private:
		using HiddenGroupList = ListInterface<HiddenGroupType>;

		template <class T>
		static T & getItem(ListInterface<T> & list, std::size_t index) {
			return list.at(index);
		}

		NodeList        inputNodes_;
		NodeList        outputNodes_;
		HiddenGroupList hiddenGroupList_;

		ListInterface<std::future<FloatList> > futureList_;
};



/* Old work, can safely be deleted.
template <
	class t_NodeType,
	template <class T> class t_ListInterface = ::ListInterface
>
class alignas(t_NodeType) NodeBuffer
{
	public:
		using NodeType      = t_NodeType;
		template <class T>
		using ListInterface = t_ListInterface<T>;

		using NodePtr      = NodeType       *;
		using ConstNodePtr = NodeType const *;

		using FloatType = typename NodeType::FloatType;

		using WeightPtr      = FloatType       *;
		using ConstWeightPtr = FloatType const *;


		NodeBuffer(std::size_t layerCount,
		           std::size_t nodesPerLayer,
		           std::size_t inputNodeCount,
		           std::size_t outputNodeCount) :
			layerCount_      {layerCount},
			nodesPerLayer_   {nodesPerLayer},
			inputNodeCount_  {inputNodeCount},
			outputNodeCount_ {outputNodeCount},
			buffer_ (generate()) {}

		std::size_t getLayerCount() const {
			return layerCount_;
		}

		std::size_t getNonTerminalLayerCount() const {
			return getLayerCount() - 1;
		}

		std::size_t getNodesPerLayer() const {
			return nodesPerLayer_;
		}

		std::size_t getInputNodeCount() const {
			return inputNodeCount_;
		}

		std::size_t getOutputNodeCount() const {
			return outputNodeCount_;
		}


		NodePtr      getNode(std::size_t layerIndex, std::size_t nodeIndex) {
			return getNodeAt(calcNodeIndex(layerIndex, nodeIndex));
		}

		ConstNodePtr getNode(std::size_t layerIndex, std::size_t nodeIndex) const {
			return getNodeAt(calcNodeIndex(layerIndex, nodeIndex));
		}


		NodePtr      getNonTerminalNode(std::size_t layerIndex,
		                                std::size_t nodeIndex) {
			return getNodeAt(calcNonTerminalNodeIndex(layerIndex, nodeIndex));
		}

		ConstNodePtr getNonTerminalNode(std::size_t layerIndex,
		                                std::size_t nodeIndex) const {
			return getNodeAt(calcNonTerminalNodeIndex(layerIndex, nodeIndex));
		}


		NodePtr      getTerminalNode(std::size_t nodeIndex) {
			return getNodeAt(calcTerminalNodeIndex(nodeIndex));
		}

		ConstNodePtr getTerminalNode(std::size_t nodeIndex) const {
			return getNodeAt(calcTerminalNodeIndex(nodeIndex));
		}


		WeightPtr      getInputNodeWeightList(std::size_t nodeIndex) {
			return getWeightAt(calcInputNodeWeightListIndex(nodeIndex));
		}

		ConstWeightPtr getInputNodeWeightList(std::size_t nodeIndex) const {
			return getWeightAt(calcInputNodeWeightListIndex(nodeIndex));
		}


	private:
		using ByteList  = ListInterface<char>;

		static constexpr std::size_t NODE_SIZE      {sizeof (NodeType)};
		static constexpr std::size_t NODE_ALIGNMENT {alignof(NodeType)};
		static constexpr std::size_t FLOAT_SIZE     {sizeof (FloatType)};
		assert(NODE_ALIGNMENT >= alignof(FloatType));

		NodePtr      getNodeAt(std::size_t arrayIndex) {
			return reinterpret_cast<NodePtr>     (buffer_.at(arrayIndex));
		}

		ConstNodePtr getNodeAt(std::size_t arrayIndex) const {
			return reinterpret_cast<ConstNodePtr>(buffer_.at(arrayIndex));
		}


		WeightPtr      getWeightAt(std::size_t arrayIndex) {
			return reinterpret_cast<WeightPtr>     (buffer_.at(arrayIndex));
		}

		ConstWeightPtr getWeightAt(std::size_t arrayIndex) const {
			return reinterpret_cast<ConstWeightPtr>(buffer_.at(arrayIndex));
		}


		std::size_t calcNodeIndex(std::size_t layerIndex,
		                          std::size_t nodeIndex) const {
			if (layerIndex < getNonTerminalLayerCount())
				return calcNonTerminalNodeIndex(layerIndex, nodeIndex);
			else
				return calcTerminalNodeIndex(nodeIndex);
		}

		std::size_t calcInputNodeWeightListIndex(std::size_t nodeIndex) const {
			return nodeIndex * getInputWeightListSize();
		}

		std::size_t getFirstNonTerminalNodeIndex() const {
			return calcInputNodeWeightListIndex(getInputNodeCount());
		}

		std::size_t calcNonTerminalNodeIndex(std::size_t layerIndex,
		                                     std::size_t nodeIndex) const {
			return getFirstNonTerminalNodeIndex() +
					   layerIndex * getNonTerminalLayerSize() +
				     nodeIndex  * getNonTerminalNodeSize();
		}

		std::size_t getFirstTerminalNodeIndex() const {
			return calcNonTerminalNodeIndex(
				getNonTerminalLayerCount(), 0
			);
		}

		std::size_t calcTerminalNodeIndex(std::size_t nodeIndex) const {
			return getFirstTerminalNodeIndex() +
				     nodeIndex * getTerminalNodeSize();
		}


		std::size_t getTotalSize() const {
			return getInputWeightLayerSize() +
				     getNonTerminalGroupSize() +
					   getTerminalLayerSize();
		}

		std::size_t getInputWeightLayerSize() const {
			return getInputWeightListSize() * getInputNodeCount();
		}

		std::size_t getInputWeightListSize() const {
			return calcWeightSize(getNodesPerLayer());
		}


		std::size_t getNonTerminalGroupSize() const {
			return getNonTerminalLayerSize() * getNonTerminalLayerCount();
		}

		std::size_t getNonTerminalLayerSize() const {
			return getNonTerminalNodeSize() * getNodesPerLayer();
		}

		std::size_t getNonTerminalNodeSize() const {
			return calcNodeSize(getNodesPerLayer());
		}


		std::size_t getTerminalNodeSize() const {
			return calcNodeSize(getOutputNodeCount());
		}

		std::size_t getTerminalLayerSize() const {
			return getTerminalNodeSize() * getNodesPerLayer();
		}


		static std::size_t calcNodeSize(std::size_t connectionCount) {
			return getNodeCoreSize() + calcWeightSize(connectionCount);
		}

		/// Calculates the size required for a single node, excluding connections.
		static std::size_t getNodeCoreSize() {
			return supports::roundUpToMultiple(
				NODE_SIZE, NODE_ALIGNMENT
			);
		}

		/// Calculates the size required for a given amount of connections.
		static std::size_t calcWeightSize(std::size_t connectionCount) {
			return supports::roundUpToMultiple(
				FLOAT_SIZE * connectionCount, NODE_ALIGNMENT
			);
		}


		ByteList generate() {
			ByteList buffer;

			buffer.resize(getTotalSize());

			return buffer;
		}


		std::size_t layerCount_;
		std::size_t nodesPerLayer_;
		std::size_t inputNodeCount_;
		std::size_t outputNodeCount_;

		ByteList buffer_;
};

template <
	class t_NodeType,
	template <class T> class t_ListInterface = ::ListInterface
>
class NetworkBuffer
{
	public:
		using NodeType      = t_NodeType;
		template <class T>
		using ListInterface = t_ListInterface<T>;

		using NodePtr      = NodeType       *;
		using ConstNodePtr = NodeType const *;

		using FloatType = typename NodeType::FloatType;

		NetworkBuffer(std::size_t hiddenLayerCount,
		              std::size_t inputLayerNodeCount,
		              std::size_t hiddenLayerNodeCount,
		              std::size_t outputLayerNodeCount) :
			hiddenLayerCount_     {hiddenLayerCount},
			inputLayerNodeCount_  {inputLayerNodeCount},
			hiddenLayerNodeCount_ {hiddenLayerNodeCount},
			outputLayerNodeCount_ {outputLayerNodeCount},
			buffer_ (generate()) {}

		std::size_t getHiddenLayerCount() const {
			return hiddenLayerCount_;
		}

		std::size_t getNonTerminalHiddenLayerCount() const {
			return getHiddenLayerCount() - 1;
		}

		std::size_t getInputLayerNodeCount() const {
			return inputLayerNodeCount_;
		}

		std::size_t getHiddenLayerNodeCount() const {
			return hiddenLayerNodeCount_;
		}

		std::size_t getOutputLayerNodeCount() const {
			return outputLayerNodeCount_;
		}

		std::size_t getTotalLayerCount() const {
			return getHiddenLayerCount() + 2;
		}

		NodePtr getNode(std::size_t layerIndex, std::size_t nodeIndex) {
			return getNode(calcNodeIndex(layerIndex, nodeIndex));
		}

		ConstNodePtr getInputNode(std::size_t index) const {
			return getNode(calcInputNodeIndex(index));
		}

		NodePtr      getInputNode(std::size_t index) {
			return getNode(calcInputNodeIndex(index));
		}

		ConstNodePtr getNonTerminalHiddenNode(std::size_t layerIndex,
		                                      std::size_t nodeIndex) const {
			return getNode(calcNonTerminalHiddenNodeIndex(layerIndex, nodeIndex));
		}

		NodePtr      getNonTerminalHiddenNode(std::size_t layerIndex,
		                                      std::size_t nodeIndex) {
			return getNode(calcNonTerminalHiddenNodeIndex(layerIndex, nodeIndex));
		}

		ConstNodePtr getTerminalHiddenNode(std::size_t nodeIndex) const {
			return getNode(calcTerminalHiddenNodeIndex(nodeIndex));
		}

		NodePtr      getTerminalHiddenNode(std::size_t nodeIndex) {
			return getNode(calcTerminalHiddenNodeIndex(nodeIndex));
		}

		ConstNodePtr getHiddenNode(std::size_t layerIndex,
		                           std::size_t nodeIndex) const {
			return getNode(calcHiddenNodeIndex(layerIndex, nodeIndex));
		}

		NodePtr      getHiddenNode(std::size_t layerIndex,
		                           std::size_t nodeIndex) {
			return getNode(calcHiddenNodeIndex(layerIndex, nodeIndex));
		}


	private:
		/// Data layout:
		///  Input layer    |  Hidden layers  |  Output layer
		/// {Node, Weights} | {Node, Weights} | {Node}
		/// Detailed row indicates interleaved data.

		using ByteList  = ListInterface<char>;

		static constexpr std::size_t NODE_SIZE      {sizeof (NodeType)};
		static constexpr std::size_t NODE_ALIGNMENT {alignof(NodeType)};
		static constexpr std::size_t FLOAT_SIZE     {sizeof (FloatType)};
		assert(NODE_ALIGNMENT >= alignof(FloatType));

		ConstNodePtr getNode(std::size_t arrayIndex) const {
			return reinterpret_cast<ConstNodePtr>(buffer_.at(arrayIndex));
		}

		NodePtr      getNode(std::size_t arrayIndex) {
			return reinterpret_cast<NodePtr>     (buffer_.at(arrayIndex));
		}

		std::size_t calcNodeIndex(std::size_t layerIndex,
		                          std::size_t nodeIndex) const {
			if (layerIndex == 0)
				return calcInputNodeIndex(nodeIndex);
			else if (layerIndex <= getHiddenLayerCount())
				return calcHiddenNodeIndex(layerIndex - 1, nodeIndex);
			else
				return calcOutputNodeIndex(nodeIndex);
		}

		std::size_t calcInputNodeIndex(std::size_t nodeIndex) const {
			return nodeIndex * getInputNodeSize();
		}

		std::size_t calcFirstHiddenNodeIndex() const {
			return calcInputNodeIndex(getInputLayerNodeCount());
		}

		std::size_t calcNonTerminalHiddenNodeIndex(std::size_t layerIndex,
		                                           std::size_t nodeIndex) const {
			return calcFirstHiddenNodeIndex() +
				     layerIndex * getNonTerminalHiddenLayerSize() +
				     nodeIndex  * getNonTerminalHiddenNodeSize();
		}

		std::size_t calcFirstTerminalHiddenNodeIndex() const {
			return calcNonTerminalHiddenNodeIndex(
				getNonTerminalHiddenLayerCount(), 0
			);
		}

		std::size_t calcTerminalHiddenNodeIndex(std::size_t nodeIndex) const {
			return calcFirstTerminalHiddenNodeIndex() +
				     nodeIndex * getTerminalHiddenNodeSize();
		}

		std::size_t calcHiddenNodeIndex(std::size_t layerIndex,
		                                std::size_t nodeIndex) const {
			auto terminalLayer = getHiddenLayerCount() - 1;

			if (layerIndex < terminalLayer)
				return calcNonTerminalHiddenNodeIndex(layerIndex, nodeIndex);
			else
				return calcTerminalHiddenNodeIndex(nodeIndex);
		}

		std::size_t calcFirstOutputNodeIndex() const {
			return calcFirstHiddenNodeIndex() +
		}

		std::size_t calcOutputNodeIndex(std::size_t nodeIndex) const {


		}


		std::size_t getInputNodeSize() const {
			return calcNodeSize(1) + calcWeightSize(inputLayerNodeCount_);
		}

		std::size_t getInputLayerSize() const {
			return getInputNodeSize() * inputLayerNodeCount_;
		}

		std::size_t getNonTerminalHiddenNodeSize() const {
			return calcNodeSize(1) + calcWeightSize(hiddenLayerNodeCount_);
		}

		std::size_t getNonTerminalHiddenLayerSize() const {
			return getNonTerminalHiddenNodeSize() * hiddenLayerNodeCount_;
		}

		std::size_t getNonTerminalHiddenGroupSize() const {
			return getNonTerminalHiddenLayerSize() * hiddenLayerCount_;
		}


		std::size_t getTerminalHiddenNodeSize() const {
			return calcNodeSize(1) + calcWeightSize(outputLayerNodeCount_);
		}

		std::size_t getTerminalHiddenLayerSize() const {
			return getTerminalHiddenNodeSize() * hiddenLayerNodeCount_;
		}

		std::size_t getHiddenLayersSize() const {
			return getNonTerminalHiddenGroupSize() +
			       getTerminalHiddenLayerSize();
		}


		std::size_t getOutputNodeSize() const {
			return calcNodeSize(1);
		}

		std::size_t getOutputLayerSize() const {
			return getOutputNodeSize() * outputLayerNodeCount_;
		}

		std::size_t getTotalSize() const {
			return
				getInputLayerSize() +
		    getHiddenLayersSize() +
		    getOutputLayerSize();
		}

		static std::size_t calcNodeSize(std::size_t nodeCount) {
			return supports::roundUpToMultiple(
				NODE_SIZE * nodeCount, NODE_ALIGNMENT
			);
		}

		static std::size_t calcWeightSize(std::size_t connectionCount) {
			return supports::roundUpToMultiple(
				FLOAT_SIZE * connectionCount, NODE_ALIGNMENT
			);
		}

		ByteList generate() {
			ByteList buffer;

			buffer.resize(getTotalSize());

			for (std::size_t i {0}; i < inputLayerNodeCount_; ++i) {

			}


		}


		std::size_t hiddenLayerCount_;
		std::size_t inputLayerNodeCount_;
		std::size_t hiddenLayerNodeCount_;
		std::size_t outputLayerNodeCount_;

		ByteList buffer_;
};

template <
	class t_NodeType,
	template <class T> class t_ListInterface = ::ListInterface
>
class HiddenGroupContainer
{
	public:
		using NodeType      = t_NodeType;
		using NodeReference = NodeType &;

		template <class T>
		using ListInterface = t_ListInterface<T>;

		HiddenGroupContainer(std::size_t groupCount,
		                     std::size_t layersPerGroup,
		                     std::size_t nodesPerLayer);

	private:
		using BufferType = NodeBuffer<NodeType, ListInterface>;

		ListInterface<BufferType> hiddenNodeNets_;
};

template <
	class t_NodeType,
  template <class T> class t_ListInterface = ::ListInterface
>
class Network
{
	public:
		using NodeType      = t_NodeType;
		using NodeReference = NodeType &;

		template <class T>
		using ListInterface = t_ListInterface<T>;

		Network(std::size_t hiddenLayerCount,
		        std::size_t hiddenNetCount,
		        std::size_t nodesPerHiddenLayer,
		        std::size_t inputNodeCount,
		        std::size_t outputNodeCount) :
			inputNodes_ (inputNodeCount),
			outputNodes_(outputNodeCount),
			hiddenNodeNets_(hiddenNetCount,
			                BufferType(hiddenLayerCount, nodesPerHiddenLayer,
			                           inputNodeCount, outputNodeCount)) {}


	private:
		using BufferType = NodeBuffer<NodeType, ListInterface>;

		ListInterface<NodeType>   inputNodes_;
		ListInterface<NodeType>   outputNodes_;
		ListInterface<BufferType> hiddenNodeNets_;
};
*/



		}
	}
}

#endif
