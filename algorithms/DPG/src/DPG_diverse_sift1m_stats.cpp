#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <chrono>
#include <vector>
#include <unordered_set>
#include <filesystem>

#include "kgraph.h"
#include "kgraph-data.h"

#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif

typedef KGRAPH_VALUE_TYPE value_type;
static auto read_top_list(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        std::cerr << "error when accessing top list file" << fname << " size is: " << file_size << " message: " << ec.message() << std::endl;
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
        std::cerr << "could not open " << fname << std::endl;
        perror("");
        abort();
    }

    uint32_t dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert((file_size - 4) % ((dims + 1) * 4) == 0 || !"weird file size");
    size_t n = (file_size - 4) / ((dims + 1) * 4);

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<uint32_t[]>(n * (dims + 1));
    ifstream.read(reinterpret_cast<char*>(x.get()), n * (dims + 1) * sizeof(uint32_t));
    if (!ifstream) assert(ifstream.gcount() == static_cast<int>(n * (dims + 1)) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) memmove(&x[i * dims], &x[1 + i * (dims + 1)], dims * sizeof(uint32_t));

    ifstream.close();
    return x;
}

static void compute_stats(const char* graph_file, const uint32_t feature_dims, const char* top_list_file) {
    std::cout << "Compute graph stats of " << graph_file << std::endl;

    size_t top_list_dims;
    size_t top_list_count;
    const auto all_top_list = read_top_list(top_list_file, top_list_dims, top_list_count);
    std::cout << "Load TopList from file" << top_list_file << " with " << top_list_count << " elements and k=" << top_list_dims << std::endl;

    auto kgraph = kgraph::KGraph::create(); 
    kgraph->load(graph_file);
    auto graph = kgraph->getCompactGraph();
    auto graph_size = graph.size();
    
    // compute the graph quality
    uint64_t perfect_neighbor_count = 0;
    uint64_t total_neighbor_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        // get top list of this node
        auto top_list = all_top_list.get() + n * top_list_dims;
        if(top_list_dims < edges_per_node) {
            std::cerr << "TopList for " << n << " is not long enough has " << edges_per_node << " elements has " << top_list_dims << std::endl;
            edges_per_node = (uint16_t) top_list_dims;
        }
        total_neighbor_count += edges_per_node;

        // check if every neighbor is from the perfect neighborhood
        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indizies[e];

            // find in the neighbor ini the first few elements of the top list
            for (uint32_t i = 0; i < edges_per_node; i++) {
                if(neighbor_index == top_list[i]) {
                    perfect_neighbor_count++;
                    break;
                }
            }
        }
    }
    auto perfect_neighbor_ratio = (float) perfect_neighbor_count / total_neighbor_count;
    auto avg_edge_count = (float) total_neighbor_count / graph_size;

    // compute the min, and max out degree
    uint16_t min_out =  std::numeric_limits<uint16_t>::max();
    uint16_t max_out = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        if(edges_per_node < min_out)
            min_out = edges_per_node;
        if(max_out < edges_per_node)
            max_out = edges_per_node;
    }

    // compute the min, and max in degree
    auto in_degree_count = std::vector<uint32_t>(graph_size);
    for (uint32_t n = 0; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indizies[e];
            in_degree_count[neighbor_index]++;
        }
    }

    uint32_t min_in = std::numeric_limits<uint32_t>::max();
    uint32_t max_in = 0;
    uint32_t zero_in_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto in_degree = in_degree_count[n];

        if(in_degree < min_in)
            min_in = in_degree;
        if(max_in < in_degree)
            max_in = in_degree;

        if(in_degree == 0) {
            zero_in_count++;
            std::cout << "Node " << n << " has zero incoming connections" << std::endl;
        }
    }

    std::cout << "GQ " << perfect_neighbor_ratio << ", avg degree " << avg_edge_count << ", min_out " << min_out << ", max_out " << max_out << ", min_in " << min_in << ", max_in " << max_in << ", zero in nodes " << zero_in_count << "\n" << std::endl;
}


int main(int argc, char** argv) {

    std::cout << "KGRAPH_MATRIX_ALIGN" << KGRAPH_MATRIX_ALIGN << std::endl;
    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    #ifdef __AVX__
        std::cout << "__AVX__ is set" << std::endl;
    #endif

    auto dpg_file      = R"(c:/Data/Feature/SIFT1M/dpg/ssg_dpg_L40.dpg)"; // GQ 0.64771, avg degree 60.5439, min_out 40, max_out 322, min_in 40, max_in 322, zero in nodes 0
    //auto dpg_file         = R"(c:/Data/Feature/SIFT1M/dpg/K90 it12 S20 R50 L200 recall0.996 - L50.dpg)"; // GQ 0.646523, avg degree 75.1368, min_out 50, max_out 392, min_in 50, max_in 392, zero in nodes 0
    auto top_list_file = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base_top200_p0.998.ivecs)";
    compute_stats(dpg_file, 128, top_list_file);

    std::cout << "Finished" << std::endl;
}