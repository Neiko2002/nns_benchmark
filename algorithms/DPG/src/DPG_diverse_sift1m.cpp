
#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
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

int main(int argc, char *argv[]) {
  
    auto object_file      = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
    auto kgraph_file      = R"(c:/Data/Feature/SIFT1M/dpg/ssg_kgraph_for_dpg.kgraph)";
    auto dpg_file         = R"(c:/Data/Feature/SIFT1M/dpg/ssg_dpg_L40.dpg)";

    unsigned L = 40;

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
    kgraph::KGraph *kgraph = kgraph::KGraph::create();
    kgraph->load(kgraph_file);

    auto s = std::chrono::high_resolution_clock::now();

    // start the diversification ?? ( seperately ) 
    // add the diversification part here 
    // cerr << "Start the diversification process... (angular dissimilarity )" << endl << endl;
    // kgraph->remove_near_edges(oracle, L); // here not knn's k, L is the length of NN list ( note that the true NN list length might be smaller than L) 

    cerr << "Start the diversification process... ( counting cuts )" << endl << endl;
    kgraph->diversify_by_cut(oracle, L); // here not knn's k, L is the length of NN list ( note that the true NN list length might be smaller than L) 


    // reverse the edges here 
    cerr << "Add reverse edges ..." << endl << endl;
    kgraph->add_backward_edges();

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "indexing time: " << diff.count() << "\n";

    kgraph->save(dpg_file);

    return 0;
}
