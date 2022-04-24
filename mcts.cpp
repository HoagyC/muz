#include "mcts.h"
#include vector

namespace mcts
{
    CTreeNode::CTreeNode(int latent_index,
                         int action_size,
                         std::vector<float> pol_pred,
                         CTreeNode *parent,
                         CMinMax *minmax,
                         float val_pred = 0,
                         float reward = 0,
                         int num_visits = 1) {}

    CTreeNode::~CTreeNode() {}

    void CTreeNode::update_val(float curr_val)
    {
        float nmtr;
        nmtr = this->average_val * this->num_visits + curr_val
    }
