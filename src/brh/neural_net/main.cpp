#include <iostream>
#include <random>
#include <fstream>

//#include "layered.h"
#include "activation_functions.h"

#include "dynamic/node.h"

#include "constant/network.h"
#include "constant/node.h"

using namespace brh::neural;
using namespace brh::neural::constant;

constexpr std::size_t calcImageNodeCount(std::size_t width,
                                         std::size_t height) {
	return width * height * 3;
}

template <class T>
int randomizeGroupWeights(HiddenGroup<T, ::ListInterface> & group,
                          FloatType min = 0, FloatType max = 1) {
	static std::mt19937 engine;
	static std::uniform_real_distribution<FloatType> dist {min, max};

	auto nodesPerLayer = group.getNodesPerLayer();

	auto inputNodeCount = group.getInputNodeCount();
	for (std::size_t j {0}; j < inputNodeCount; ++j) {
		for (std::size_t k {0}; k < nodesPerLayer; ++k) {
			*group.getInputWeight(j, k) = dist(engine);
		}
	}

	auto nonTerminalLayerCount = group.getNonTerminalLayerCount();
	for (std::size_t j {0}; j < nonTerminalLayerCount; ++j) {
		for (std::size_t k {0}; k < nodesPerLayer; ++k) {
			for (std::size_t l {0}; l < nodesPerLayer; ++l) {
				*group.getNonTerminalElement(j, k).getWeight(l) = dist(engine);
			}
		}
	}

	for (std::size_t j {0}; j < nodesPerLayer; ++j) {
		for (std::size_t k {0}; k < group.getOutputNodeCount(); ++k) {
			*group.getTerminalElement(j).getWeight(k) = dist(engine);
		}
	}

	return 0;
}

template <class T>
void randomizeWeights(Network<T> & network, FloatType min = 0, FloatType max = 1) {
	auto hiddenGroupCount = network.getHiddenGroupCount();
	std::vector<std::future<int> > futureList (hiddenGroupCount);

	for (std::size_t i {0}; i < hiddenGroupCount; ++i) {
		futureList[i] = std::async(
			std::launch::async, randomizeGroupWeights<T>,
			std::ref(network.getHiddenGroup(i)), min, max
		);
	}

	for (auto & i : futureList)
		i.get();
}

int main(int argc, char * argv[])
{
	//using namespace layered;

	/*LayerList layers {
		{
			generateRandomNodes(2, 3),
		  generateRandomNodes(3, 1),
			{{Node()}}
		}
	};

	Network net {std::move(layers)};

	auto & input = net.getLayers()[0];

	input.getNodes()[0].setValue(1);
	input.getNodes()[1].setValue(1);

	net.execute();

	auto & output = net.getLayers().back();

	std::cout << output.getNode(0).getValue() << '\n';

	auto net2 = generateNetwork(1, 2, 3, 1);

	net2.execute({1, 1});

	std::cout << net2.getLayers().back().getNode(0).getValue() << '\n';*/

	using Net = Network<Node, ::ListInterface>;

	/*Net net {4, 10, 10, 2};

	std::size_t index {0};
	std::size_t needed {0};

	for (std::size_t i {0}; i < net.getInputNodeCount(); ++i) {
		for (std::size_t j {0}; j < net.getNodesPerLayer(); ++j) {
			std::cout << i << ", " << j << ": " << net.getInputWeightIndex(i, j) << '\n';
			*net.getInputWeight(i, j) = index++;
		}
	}

	needed += net.getInputLayerSize();
	std::cout << needed << ' ' << index << '\n';

	for (std::size_t i {0}; i < net.getNonTerminalLayerCount(); ++i) {
		for (std::size_t j {0}; j < net.getNodesPerLayer(); ++j) {
			auto ptr = net.getNonTerminalElement(i, j);
			ptr->setValue(index++);
			std::cout << i << ", " << j << ": " << net.getNonTerminalElementIndex(i, j) << '\n';
			for (std::size_t k {0}; k < net.getNodesPerLayer(); ++k) {
				std::cout << k << ": " << index << '\n';
				*ptr->getWeight(k) = index++;
			}
		}
	}

	needed += net.getNonTerminalGroupSize();
	std::cout << needed << ' ' << index << '\n';

	for (std::size_t i {0}; i < net.getNodesPerLayer(); ++i) {
		auto ptr = net.getTerminalElement(i);
		ptr->setValue(index++);
		std::cout << i << ": " << net.getTerminalElementIndex(i) << '\n';
		for (std::size_t j {0}; j < net.getOutputNodeCount(); ++j) {
			std::cout << i << ',' << j << '\n';
			*ptr->getWeight(j) = index++;
		}
	}

	needed += net.getTerminalLayerSize();
	std::cout << needed << ' ' << index << '\n';*/

	/*Net net (2, 1, 1, 2, 1);
	randomizeWeights(net);

	net.getInputNode(0).setValue(0.1);

	for (std::size_t i {0}; i < net.getHiddenGroupCount(); ++i) {
		auto & group = net.getHiddenGroup(i);

		*group.getInputWeight(0, 0) = 0.5;
		*group.getNonTerminalElement(0, 0).getWeight(0) = .1;
		*group.getTerminalElement(0).getWeight(0) = 1;
	}*/

	/*std::vector<Node> inputNodes;
	inputNodes.reserve(1);
	inputNodes.emplace_back();
	inputNodes.back().setValue(.1);

	*net.getInputWeight(0, 0) = 0.5;

	*net.getNonTerminalElement(0, 0)->getWeight(0) = .1;
	*net.getTerminalElement(0)->getWeight(0) = 1;

	auto out = net.execute(inputNodes.data(), softStep);

	Node outNode;
	outNode.applyActivation(softStep);

	std::cout << outNode.getValue() << '\n';*/

	//net.execute(softStep);

	constexpr std::size_t WIDTH  {100};
	constexpr std::size_t HEIGHT {100};
	constexpr std::size_t NODE_COUNT {calcImageNodeCount(WIDTH, HEIGHT)};

	std::ifstream inFile ("image1.data", std::ios::binary);

	Net bigNet (2, NODE_COUNT, NODE_COUNT, 10, 10);
	randomizeWeights(bigNet, 0, .5);

	{
		std::size_t i {0};
		char currentChar;
		while (inFile.get(currentChar) && i < bigNet.getInputNodeCount()) {
			bigNet.getInputNode(i).setValue(static_cast<float>(static_cast<unsigned char>(currentChar)) / 256);
			++i;
		}
	}

	bigNet.execute(softStep);

	std::ofstream outFile("image1_out.data", std::ios::binary);

	for (std::size_t i {0}; i < bigNet.getOutputNodeCount(); ++i) {
		std::cout << std::round(bigNet.getOutputNode(i).getValue() * 256) << '\n';
		outFile.put(static_cast<char>(std::round(bigNet.getOutputNode(i).getValue() * 256)));
	}

	return 0;
}