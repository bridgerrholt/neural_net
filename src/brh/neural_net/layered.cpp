#include "layered.h"

#include <cassert>
#include <cmath>
#include <random>

namespace layered {

Node::Node() : Node(ListType()) {}
Node::Node(ListType weights) : weights_ (std::move(weights)), value_ {0} {}

void Node::clearValue() { value_ = 0; }

void Node::setValue(FloatType value) { value_ = value; }

void Node::addValue(FloatType value) { value_ += value; }

FloatType Node::getValue() const { return value_; }

void Node::applyActivation()
{
	value_ = static_cast<FloatType>(1.0 / (1 + std::pow(MATH_E, -value_)));
}

FloatType Node::getWeighted(std::size_t index) const
{
	return value_ * weights_.at(index);
}

ListType const & Node::getWeights() const { return weights_; }
ListType       & Node::getWeights()       { return weights_; }



Layer::Layer(NodeList nodes) : nodes_ (std::move(nodes)) {}


void Layer::clearValues()
{
	for (auto & i : nodes_)
		i.clearValue();
}


void Layer::propagate(Layer & nextLayer)
{
	auto thisSize = nodes_.size();
	auto nextSize = nextLayer.nodes_.size();
	for (std::size_t i {0}; i < thisSize; ++i) {
		assert(nodes_[i].getWeights().size() == nextSize);

		for (std::size_t j {0}; j < nextSize; ++j) {
			nextLayer.nodes_[j].addValue(nodes_[i].getWeighted(j));
		}
	}
}

void Layer::applyActivation()
{
	for (auto & i : nodes_)
		i.applyActivation();
}

NodeList const & Layer::getNodes() const { return nodes_; }
NodeList       & Layer::getNodes()       { return nodes_; }


Node& Layer::getNode(std::size_t index) { return nodes_.at(index); }



Network::Network(LayerList layers) : layers_ (std::move(layers)) {}

void Network::execute(ListType inputValues)
{
	auto & nodeList = layers_[0].getNodes();

	auto size = inputValues.size();

	assert(size == nodeList.size());

	for (std::size_t i {0}; i < size; ++i) {
		nodeList[i].setValue(inputValues[i]);
	}

	execute();
}

void Network::execute()
{
	for (std::size_t i {1}; i < layers_.size(); ++i) {
		auto & previous = layers_[i - 1];
		auto & current  = layers_[i];

		current.clearValues();
		previous.propagate(current);
		current.applyActivation();
	}

	/*auto & previous = *(layers_.end() - 2);
	auto & current  =   layers_.back();

	current.clearValues();
	previous.propagate(current);*/
}

LayerList const & Network::getLayers() const { return layers_; }
LayerList       & Network::getLayers()       { return layers_; }


NodeList generateRandomNodes(std::size_t count, std::size_t connectionCount)
{
	static std::mt19937 randEngine;
	std::uniform_real_distribution<FloatType> dist {0, 1};

	NodeList nodes;

	for (std::size_t i {0}; i < count; ++i) {
		ListType weights;
		weights.resize(connectionCount);

		for (std::size_t j {0}; j < connectionCount; ++j) {
			weights[j] = dist(randEngine);
		}

		nodes.emplace_back(std::move(weights));
	}

	return nodes;
}


Network generateNetwork(std::size_t hiddenLayerCount,
                        std::size_t inputLayerSize,
                        std::size_t hiddenLayerSize,
                        std::size_t outputLayerSize)
{
	constexpr std::size_t inputLayerIndex {0};
	constexpr std::size_t hiddenLayersBegin {1};

	std::size_t const hiddenLayersEnd  {hiddenLayersBegin + hiddenLayerCount};
	std::size_t const outputLayerIndex {hiddenLayersEnd};

	std::size_t const layerCount {outputLayerIndex + 1};

	LayerList layers;
	layers.reserve(layerCount);

	layers.emplace_back(generateRandomNodes(inputLayerSize, hiddenLayerSize));

	for (std::size_t i {0}; i < hiddenLayerCount - 1; ++i) {
		layers.emplace_back(
			generateRandomNodes(hiddenLayerSize, hiddenLayerSize)
		);
	}

	layers.emplace_back(generateRandomNodes(hiddenLayerSize, outputLayerSize));

	NodeList outputNodes (outputLayerSize);
	layers.emplace_back(std::move(outputNodes));

	return {std::move(layers)};
}

}