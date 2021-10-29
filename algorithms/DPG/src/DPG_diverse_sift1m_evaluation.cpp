#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <chrono>
#include <vector>
#include <unordered_set>

#include "kgraph.h"
#include "kgraph-data.h"

#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif

typedef KGRAPH_VALUE_TYPE value_type;

static void load_data(const char* filename, float*& data, unsigned& num,unsigned& dim) {
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
    in.seekg(4,std::ios::cur);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
}

static void load_data(const char *object_file, kgraph::Matrix<value_type>& data) {
    float* data_load = NULL;
    unsigned points_num, dim;
    load_data(object_file, data_load, points_num, dim);

    // copy the data into an aligned matrix
    data.resize(points_num, dim);
    for (unsigned i = 0; i < points_num; ++i) {
        value_type *row = data[i];
        for (unsigned j = 0; j < dim; ++j) 
            row[j] = data_load[i * dim + j];
    }
    delete data_load;
}

static std::vector<std::unordered_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const size_t k)
{
    auto answers = std::vector<std::unordered_set<uint32_t>>();
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++)
    {
        auto gt = std::unordered_set<uint32_t>();
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) gt.insert(ground_truth[k * i + j]);

        answers.push_back(gt);
    }

    return answers;
}

int main (int argc, char *argv[]) {

    std::cout << "KGRAPH_MATRIX_ALIGN" << KGRAPH_MATRIX_ALIGN << std::endl;
    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    auto object_file      = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
    auto dpg_file         = R"(c:/Data/Feature/SIFT1M/dpg/ssg_dpg_L40.dpg)";
    //auto dpg_file         = R"(c:/Data/Feature/SIFT1M/dpg/K90 it12 S20 R50 L200 recall0.996 - L50.dpg)";
    auto query_file       = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_query.fvecs)";
    auto groundtruth_file = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_groundtruth.ivecs)";
    auto k = 100;

    // expert options
    // https://github.com/Neiko2002/kgraph/blob/master/doc/params.md
    kgraph::KGraph::SearchParams params;
    params.K = k;
    params.M = kgraph::default_M; // Use default.
    params.P = kgraph::default_P; // main parameter to control the amount of computation. 
    params.T = kgraph::default_T; // repreat T times
    params.init = 0;

    // database features 
    kgraph::Matrix<value_type> data;
    load_data(object_file, data);
    kgraph::MatrixOracle<value_type, kgraph::metric::l2sqr> oracle(data);
  
    // load the graph
    kgraph::KGraph *kgraph = kgraph::KGraph::create(); 
    kgraph->load(dpg_file);

    // the query data 
    kgraph::Matrix<value_type> queries;
    load_data(query_file, queries);

    // query ground truth
    float* groundtruth_f = NULL;
    unsigned groundtruth_num, groundtruth_dim;
    load_data(groundtruth_file, groundtruth_f, groundtruth_num, groundtruth_dim);
    const auto ground_truth = (uint32_t*)groundtruth_f; // not very clean, works as long as sizeof(int) == sizeof(float)
    const auto answers = get_ground_truth(ground_truth, groundtruth_num, k);

    // try differen P_search parameters
    auto result = kgraph::Matrix<unsigned>(queries.size(), params.K);
    std::vector<unsigned> P_search_parameter = { 3, 10, 50, 100 };
    for (unsigned P_search : P_search_parameter) {
      params.P = P_search;

      // search the graph
      auto time_begin = std::chrono::steady_clock::now();
      for (int64_t i = 0; i < queries.size(); ++i) 
        kgraph->search(oracle.query(queries[i]), params, result[i], nullptr);

      // compare
      size_t correct = 0;
      for (size_t i = 0; i < queries.size(); i++) {
        auto answer = answers[i];
        auto predictions = result[i];
        for (size_t r = 0; r < k; r++)
          if (answer.find(predictions[r]) != answer.end()) correct++;
      }
      auto recall = 1.0f * correct / (queries.size() * k);

      auto time_end = std::chrono::steady_clock::now();
      auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / queries.size();
      std::cout << "P_search " << P_search << ", recall " << recall << " time_us_per_query " << time_us_per_query << std::endl;
    }
    
    return 0;
}