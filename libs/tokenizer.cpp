#include "./tokenizer.hpp"

#include <ostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

Tokenizer::Tokenizer() {
    for (int i = 0; i < 256; ++i) {
        vocab[i] = {static_cast<uint8_t>(i)};
    }
}

std::vector<int> Tokenizer::getCodePoints(const std::string& data) {
    std::vector<int> codePoints;
    codePoints.reserve(data.size());
    for (unsigned char c : data) {
        codePoints.push_back(static_cast<int>(c));
    }
    return codePoints;
}

std::unordered_map<std::pair<int, int>, int, pair_hash> Tokenizer::getPairs(const std::vector<int>& tokens) {
    std::unordered_map<std::pair<int, int>, int, pair_hash> pairs;
    if (tokens.size() < 2) return pairs;
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        pairs[{tokens[i], tokens[i + 1]}]++;
    }
    return pairs;
}

std::vector<int> Tokenizer::doMerge(const std::vector<int>& tokens, int size, const std::pair<int, int>& pair, int idx) {
    std::vector<int> newTokens;
    newTokens.reserve(size);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i < tokens.size() - 1 && tokens[i] == pair.first && tokens[i + 1] == pair.second) {
            newTokens.push_back(idx);
            i++; 
        } else {
            newTokens.push_back(tokens[i]);
        }
    }
    return newTokens;
}

void Tokenizer::train(const std::string& filename, int nVocab) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string data = buffer.str();

    int cycles = nVocab - 256;
    std::vector<int> tokens = getCodePoints(data);
    for (int i = 0; i < cycles; ++i) {
        auto pairs = getPairs(tokens);
        if (pairs.empty()) break;

        auto mostFreqPair = std::max_element(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        int newIdx = 256 + i;
        tokens = doMerge(tokens, tokens.size() - mostFreqPair->second, mostFreqPair->first, newIdx);
        mergeForest[mostFreqPair->first] = newIdx;

        std::cout << "Updated Total Tokens: " << tokens.size() << ", Merging (" << mostFreqPair->first.first << ", " << mostFreqPair->first.second << ") into " << newIdx << std::endl;

        std::vector<uint8_t> newBytes = vocab[mostFreqPair->first.first];
        newBytes.insert(newBytes.end(), vocab[mostFreqPair->first.second].begin(), vocab[mostFreqPair->first.second].end());
        vocab[newIdx] = newBytes;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens = getCodePoints(text);
    while (true) {
        auto pairs = getPairs(tokens);
        std::pair<int, int> bestPair = {-1, -1};
        int pairFreq = 0;
        int minMergeIdx = 2e9; 

        for (const auto& p : pairs) {
            auto it = mergeForest.find(p.first);
            if (it != mergeForest.end() && it->second < minMergeIdx) {
                minMergeIdx = it->second;
                bestPair = p.first;
                pairFreq = p.second;
            }
        }

        if (bestPair.first == -1) break;
        tokens = doMerge(tokens, tokens.size() - pairFreq, bestPair, mergeForest[bestPair]);
    }
    return tokens;
}

py::bytes Tokenizer::decode(const std::vector<int>& tokens) {
    std::vector<uint8_t> decodedBytes;
    for (int idx : tokens) {
        auto it = vocab.find(idx);
        if (it != vocab.end()) {
            decodedBytes.insert(decodedBytes.end(), it->second.begin(), it->second.end());
        }
    }
    return py::bytes(std::string(decodedBytes.begin(), decodedBytes.end()));
}
