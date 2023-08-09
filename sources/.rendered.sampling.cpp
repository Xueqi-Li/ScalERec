#include </home/lixueqi/anaconda3/envs/py38/lib/python3.11/site-packages/pybind11/include/pybind11/pybind11.h>
// #include <pybind11/pybind11.h>
#include </home/lixueqi/anaconda3/envs/py38/lib/python3.11/site-packages/pybind11/include/pybind11/stl.h>
#include </home/lixueqi/anaconda3/envs/py38/lib/python3.11/site-packages/pybind11/include/pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

int randint_(int end)
{
    return rand() % end;
}

py::array_t<int> sample_negative(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int perUserNum = (train_num / user_num);
    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    // py::array_t<int> S_array = py::array_t<int>({train_num, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user = 0; user < user_num; user++)
    {
        std::vector<int> pos_item = allPos[user];
        if (pos_item.size() == 0)
        {
            continue;
        }
        for (int pair_i = 0; pair_i < perUserNum; pair_i++)
        {
            int negitem = 0;
            ptr[(user * perUserNum + pair_i) * row] = user;
            ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];
            for (int index = 2; index < neg_num + 2; index++)
            {
                do
                {
                    negitem = randint_(item_num);
                } while (
                    find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
                ptr[(user * perUserNum + pair_i) * row + index] = negitem;
            }
        }
    }
    return S_array;
}

py::array_t<int> sample_negative_subg(std::vector<int> users, std::vector<int> items, int train_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    // BPR sampling for subgraph-wise sampling GNN, keeping user, pos, neg within graph
    int user_num = users.size();
    int perUserNum = (train_num / user_num);
    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user_i = 0; user_i < user_num; user_i++)
    {
        // cout << user_i << endl;
        int user = users[user_i];
        std::vector<int> pos_item = allPos[user];
        if (pos_item.size() == 0)
        {
            continue;
        }
        for (int pair_i = 0; pair_i < perUserNum; pair_i++)
        {
            int pos_item_u = 0;
            int counter = 0;
            int flag_c = 0;
            do
            {
                pos_item_u = pos_item[randint_(pos_item.size())];
                counter++;
                if (counter > 1000)
                {
                    // cout << user << "," << pair_i << ", SKIP. hard pos sampling!" << endl;
                    flag_c = -1;
                }
            } while (
                find(items.begin(), items.end(), pos_item_u) == items.end() &&
                (flag_c == 0));
            if (flag_c == -1)
            {
                continue;
            }

            ptr[(user_i * perUserNum + pair_i) * row] = user;
            ptr[(user_i * perUserNum + pair_i) * row + 1] = pos_item_u;
            // ptr[(user_i * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];

            int negitem = 0;
            for (int index = 2; index < neg_num + 2; index++)
            {
                do
                {
                    negitem = items[randint_(items.size())];
                } while (
                    find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
                ptr[(user_i * perUserNum + pair_i) * row + index] = negitem;
            }
        }
        // cout << user_i << endl;
    }
    return S_array;
}

py::array_t<int> sample_negative_ppr(int user_num, int item_num, int train_num, float ppr_neg, std::vector<std::vector<int>> allPos, std::vector<std::vector<int>> ppr_idx, int neg_num)
{
    int perUserNum = (train_num / user_num);
    int PPRNum = (perUserNum * ppr_neg);
    int PPRNum_u = 0;
    int row = neg_num + 2;
    bool flag_ppr = true;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    // py::array_t<int> S_array = py::array_t<int>({train_num, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user = 0; user < user_num; user++)
    {
        std::vector<int> pos_item = allPos[user];
        std::vector<int> ppr_u = ppr_idx[user];
        if (ppr_u.size() == 0)
        {
            PPRNum_u = 0;
            // cout << user << ", no ppr" << endl;
        }
        else
            PPRNum_u = PPRNum;
        if (pos_item.size() == 0)
        {
            continue;
        }
        for (int pair_i = 0; pair_i < perUserNum; pair_i++)
        {
            int negitem = 0;
            ptr[(user * perUserNum + pair_i) * row] = user;
            ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];
            for (int index = 0; index < neg_num; index++)
            {
                do
                {
                    if (index < PPRNum_u)
                    {
                        negitem = ppr_u[randint_(ppr_u.size())];
                        flag_ppr = true;
                    }
                    else
                    {
                        negitem = randint_(item_num);
                        flag_ppr = false;
                    }
                } while (
                    find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
                ptr[(user * perUserNum + pair_i) * row + index + 2] = negitem;
                if (flag_ppr == true)
                    PPRNum_u -= 1;
            }
        }
    }
    return S_array;
}

py::array_t<int> sample_negative_ByUser(std::vector<int> users, int item_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int row = neg_num + 2;
    int col = users.size();
    py::array_t<int> S_array = py::array_t<int>({col, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user_i = 0; user_i < users.size(); user_i++)
    {
        int user = users[user_i];
        std::vector<int> pos_item = allPos[user];
        int negitem = 0;

        ptr[user_i * row] = user;
        ptr[user_i * row + 1] = pos_item[randint_(pos_item.size())];

        for (int neg_i = 2; neg_i < row; neg_i++)
        {
            do
            {
                negitem = randint_(item_num);
            } while (
                find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
            ptr[user_i * row + neg_i] = negitem;
        }
    }
    return S_array;
}

void set_seed(unsigned int seed)
{
    // cout << seed << endl;
    srand(seed);
}

using namespace py::literals;

PYBIND11_MODULE(sampling, m)
{
    srand(time(0));
    // srand(2020);
    m.doc() = "example plugin";
    m.def("randint", &randint_, "generate int between [0 end]", "end"_a);
    m.def("seed", &set_seed, "set random seed", "seed"_a);
    m.def("sample_negative", &sample_negative, "sampling negatives for all", "user_num"_a, "item_num"_a, "train_num"_a, "allPos"_a, "neg_num"_a);
    m.def("sample_negative_ppr", &sample_negative_ppr, "sampling negatives based on ppr for all",
          "user_num"_a, "item_num"_a, "train_num"_a, "ppr_neg"_a, "allPos"_a, "ppr_idx"_a, "neg_num"_a);
    m.def("sample_negative_subg", &sample_negative_subg, "sampling negatives based on ppr for all",
          "users"_a, "items"_a, "train_num"_a, "allPos"_a, "neg_num"_a);
    m.def("sample_negative_ByUser", &sample_negative_ByUser, "sampling negatives for given users",
          "users"_a, "item_num"_a, "allPos"_a, "neg_num"_a);
}
