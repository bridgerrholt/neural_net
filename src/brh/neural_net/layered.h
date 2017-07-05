#ifndef NEURAL_NET_TESTING_LAYERED_H
#define NEURAL_NET_TESTING_LAYERED_H

#include "common.h"

namespace layered {

class Node
{
	public:
		Node();
		Node(ListType weights);

		void clearValue();
		void setValue(FloatType value);
		void addValue(FloatType value);
		FloatType getValue() const;

		void applyActivation();

		FloatType getWeighted(std::size_t index) const;

		ListType const & getWeights() const;
		ListType       & getWeights();


	private:
		ListType  weights_;
		FloatType value_;
};

using NodeList = std::vector<Node>;


class Layer
{
	public:
		Layer(NodeList nodes);

		void clearValues();

		void propagate(Layer & nextLayer);
		void applyActivation();

		NodeList const & getNodes() const;
		NodeList       & getNodes();

		Node & getNode(std::size_t index);


	private:
		NodeList nodes_;
};

using LayerList = std::vector<Layer>;


class Network
{
	public:
		Network(LayerList layers);

		void execute(ListType inputValues);
		void execute();

		LayerList const & getLayers() const;
		LayerList       & getLayers();

	private:
		LayerList layers_;
};


/// @param connectionCount How many connections each node has.
NodeList generateRandomNodes(std::size_t count, std::size_t connectionCount);

Network generateNetwork(std::size_t hiddenLayerCount,
                        std::size_t inputLayerSize,
                        std::size_t hiddenLayerSize,
                        std::size_t outputLayerSize);

}

#endif
