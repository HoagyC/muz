#ifndef MCTS_H
#define MCTS_H

#include <vector>
#include <stdlib.h>

namespace mcts
{
    class CTreeNode
    {
    public:
        int action_size, latent_index;
        float val_pred, reward_pred;
        std::vector<int> pol_pred;

        CTreeNode();
        CTreeNode(int latent_index,
                  int action_size,
                  std::vector<float> pol_pred,
                  CTreeNode *parent,
                  CMinMax *minmax,
                  float reward,
                  int num_visits,
                  float val_pred);
        ~CTreeNode();

        void update_val(float curr_val);
        float action_score(int action_n, int total_visit_count) int pick_action() int pick_game_action(float temperature);

        CTreeNode *insert();
    };

    class CMinMax
    {
    public:
        CMinMax();
        ~CMinMax();

        void update(float val);
        float normalize(float val);
    };
};

#endif