#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>


#include "stc_ml_c.h"

std::tuple<bool,
           std::vector<int>,
           std::array<unsigned int,2>>
    embed(std::vector<int> cover,
          std::vector<float> prob_map,
          std::vector<unsigned char> message)
    
    // Will return (True, [...], {x,x}) if embeding is successfull
    //             (False, [...], {x,x}) otherwise

{       
    const uint n = cover.size(); // cover size
    uint h = 10;                 // constraint height of STC code
    uint m = message.size();     // number of message bits to embed
    uint trials = 1;             // if the message cannot be embedded due to large amount of 
                                 // wet pixels, then try again with smaller message. Try at most 10 times.

    std::vector<int> stego(n);
    std::vector<unsigned char> extracted_message(m);
    std::array<unsigned int,2> num_msg_bits = {0,0};

    float* costs = new float[n*3];
    for (uint i=0; i<n; i++) {
        if (*(cover.data()+i)==0) { // F_INF is defined as infinity in float
            costs[3*i+0] = F_INF;   // cost of changing pixel by -1 is infinity => change to -1 is forbidden
            costs[3*i+1] = 0;                       // cost of changing pixel by  0
            costs[3*i+2] = log(2/prob_map[i]-2);    // cost of changing pixel by +1
        } else if (*(cover.data()+i)==255) {
            costs[3*i+0] = log(2/prob_map[i]-2);    // cost of changing pixel by -1 
            costs[3*i+1] = 0;                       // cost of changing pixel by  0
            costs[3*i+2] = F_INF;   // cost of changing pixel by +1 is infinity => change to -1 is forbidden
        } else {
            costs[3*i+0] = log(2/prob_map[i]-2);    // cost of changing pixel by -1
            costs[3*i+1] = 0;                       // cost of changing pixel by  0
            costs[3*i+2] = log(2/prob_map[i]-2);    // cost of changing pixel by +1
        }
    }

    stc_pm1_pls_embed(n, cover.data(), costs, m, message.data(),
                      h, F_INF, stego.data(), num_msg_bits.data(), trials, 0);

    stc_ml_extract(n, stego.data(), 2, num_msg_bits.data(), h, extracted_message.data()); 

    delete[] costs;

    bool msg_ok = true;
    for (uint i=0; i<m; i++) {
        msg_ok &= (*(extracted_message.data()+i)==*(message.data()+i));
    }

    return std::make_tuple(msg_ok,stego,num_msg_bits);
}

std::vector<unsigned char> extract(std::vector<int> stego,
                                   std::array<unsigned int, 2> num_msg_bits) {

    const uint n = stego.size(); // stego size
    uint h = 10;               // constraint height of STC code

    std::vector<unsigned char> extracted_message(num_msg_bits[0]+num_msg_bits[1]);

    stc_ml_extract(n, stego.data(), 2, num_msg_bits.data(), h, extracted_message.data());    

    return extracted_message;
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(STC, m) {
    m.def("embed", &embed);
    m.def("extract",&extract);
}