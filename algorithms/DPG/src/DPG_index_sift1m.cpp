#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include "kgraph.h"
#include "kgraph-data.h"

typedef KGRAPH_VALUE_TYPE value_type;

void load_data(const char* filename, float*& data, unsigned& num,unsigned& dim) {
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

int main (int argc, char *argv[]) {

    std::cout << "KGRAPH_MATRIX_ALIGN" << KGRAPH_MATRIX_ALIGN << std::endl;
    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    auto object_file      = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
    auto kgraph_file      = R"(c:/Data/Feature/SIFT1M/dpg/ssg_kgraph_R100.kgraph)";


    kgraph::KGraph::IndexParams params;

    // general options
    params.K = kgraph::default_K; // "number of nearest neighbor"
    params.controls = kgraph::default_controls; // "number of control pounsigneds"

    // expert options
    // https://github.com/Neiko2002/kgraph/blob/master/doc/params.md
    params.iterations = kgraph::default_iterations;
    params.S = kgraph::default_S;
    params.R = kgraph::default_R;
    params.L = kgraph::default_L;
    params.delta = kgraph::default_delta;
    params.recall = kgraph::default_recall;
    params.prune = kgraph::default_prune;
    params.seed = kgraph::default_seed;

    // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters 
    // K=90 it=12 S=20 R=50 L=130
    // https://github.com/Neiko2002/nns_benchmark/tree/windows-compat/algorithms/DPG
    // L need to be double the size of the DPG L
    /*params.K = 90;
    params.iterations = 12;
    params.S = 20;
    params.R = 50;
    params.L = 130;
    params.recall = 1.00;*/

    // https://arxiv.org/pdf/1907.06146v3.pdf
    params.K = 400;
    params.iterations = 12;
    params.S = 20;
    params.R = 100; 
    params.L = 500;
    params.recall = 1.00;

    // database features 
    float* data_load = NULL;
    unsigned points_num, dim;
    load_data(object_file, data_load, points_num, dim);

    // copy the data into an aligned matrix
    auto data = kgraph::Matrix<value_type>(points_num, dim);
    for (unsigned i = 0; i < points_num; ++i) {
        value_type *row = data[i];
        for (unsigned j = 0; j < dim; ++j) 
            row[j] = data_load[i * dim + j];
    }

    kgraph::MatrixOracle<value_type, kgraph::metric::l2sqr> oracle(data);
    kgraph::KGraph::IndexInfo info;
    kgraph::KGraph *kgraph = kgraph::KGraph::create(); 
    // IMPORTANT: Note that we modify the index save procedure to reduce index size    
    kgraph->build(oracle, params, kgraph_file, &info);
    std::cerr << info.stop_condition << std::endl;

    return 0;
}