/*
 * @file main.cpp
 *
 * @author Andreas Lydakis andlydakis@gmail.com
 * @date 9/26/2017
 * @brief MDP Solver that uses a variety of algorithm
 *
 */
#include <iostream>
#include <set>
#include "/usr/local/lib/cxxopts/include/cxxopts.hpp"
#include "csv.h"
#include "gurobi_c++.h"

int main(int argc, char *argv[]) {
    cxxopts::Options options(argv[0], " - example command line options");
    options.add_options()
            ("f,file", "File name", cxxopts::value<std::string>()->default_value("./samples.csv"), "")
            ("i, iterations", "Number of iterations", cxxopts::value<int>()
                    ->default_value("1000"), "N")
            ("r, runs", "Number of runs", cxxopts::value<int>()
                    ->default_value("1000"), "R")
            ("t, threshold", "Ending threshold", cxxopts::value<double>()
                    ->default_value("0.00001"), "T")
            ("g, gamma", "Gamma", cxxopts::value<double>()
                    ->default_value("0.9"), "G")
            ("a, alpha", "Alpha", cxxopts::value<double>()
                    ->default_value("0.001"), "A")
            ("m, method", "Algorithm to use", cxxopts::value<std::string>()
                    ->default_value("TD0"), "M");

    enum methods {
        LP, VI, PI, MPI, TD0, EVMC
    };
    options.parse(argc, argv);
    std::cout << "Parsed Args" << std::endl;
    std::string file = options["file"].as<std::string>();
    double gamma = options["gamma"].as<double>();
    double alpha = options["alpha"].as<double>();
    double threshold = options["threshold"].as<double>();
    int iter = options["iterations"].as<int>();
    std::string m = options["method"].as<std::string>();
    int method = LP;
    if (m == "LP") {
        method = LP;
    } else if (m == "VI") {
        method = VI;
    } else if (m == "PI") {
        method = PI;
    } else if (m == "MPI") {
        method = MPI;
    } else if (m == "TD0") {
        method = TD0;
    } else if (m == "EVMC") {
        method = EVMC;
    }
    std::cout << "Reading: " << file << std::endl;

    if (method == TD0) {
        int step, inv, ord;
        int cur_inv, cur_ord;
        double rew;
        double cur_rew;
        std::string sim;
        io::CSVReader<5> in(file);
        in.read_header(io::ignore_extra_column, "Step", "Inventory", "Order", "Reward", "Simulation");
        in.read_row(step, cur_inv, cur_ord, cur_rew, sim);
        int n_states = cur_inv;
        int n_actions = std::stoi(sim.substr(sim.find('=') + 1)) + 1;
        double V[n_actions] = {0.0};
        while (in.read_row(step, inv, ord, rew, sim)) {
            if (step != 1) {
                std::cout << step << " " << inv << " " << ord << " " << rew << " " << sim << " " << std::endl;
                V[cur_inv] += alpha * (cur_rew + gamma * V[inv] - V[cur_inv]);
                if (inv > n_states) {
                    n_states = inv;
                }
            }
            cur_inv = inv;
            cur_rew = rew;
            cur_ord = ord;
        }
        std::cout << n_actions << " States" << std::endl;
        std::cout << n_actions << " Actions" << std::endl;
        std::cout << "Gamma: " << gamma << std::endl;
        std::cout << "Alpha: " << alpha << std::endl;
        for (unsigned int i = 0; i < n_actions; i++) {
            std::cout << "v[s_" << i << "]: " << V[i] << std::endl;
        }
    } else if (method == EVMC) {
        int n_actions = 0;

        int step, inv, ord;
        int num_episodes = 0;
        double rew;
        std::string sim;

        typedef std::pair<int, int> State_Action;
        typedef std::pair<State_Action, double> Ep_Step;
        typedef std::vector<Ep_Step> Episode;
        std::map<int, double> values;
        std::vector<Episode> episodes;

        io::CSVReader<5> in(file);
//    Step,Inventory,Order,Reward,Simulation
        in.read_header(io::ignore_extra_column, "Step", "Inventory", "Order", "Reward", "Simulation");

        Episode *ce;
        while (in.read_row(step, inv, ord, rew, sim)) {
            if (step == 1) {
                num_episodes++;
                Episode ep_;
                episodes.emplace_back(ep_);
                ce = &episodes.back();
            }
            State_Action sa = State_Action(inv, ord);
            ce->emplace_back(Ep_Step(sa, rew));
//            std::cout << step << " " << sa.first << " " << sa.second << " " << rew << " " << sim << " " << std::endl;
            if (ord > n_actions) n_actions = ord;
        }
        std::cout << episodes.size() << " episodes" << std::endl;
        n_actions += 1;
        int n_states = n_actions;
        int episode_n = 0;
        double V[n_states] = {0.0};
        auto episode_it = episodes.begin();
        while (episode_it != episodes.end()) {
            episode_n++;
            auto current_episode = *episode_it;
            auto current_episode_it = current_episode.begin();
            current_episode_it++;
            std::cout << "Episode Size: " << current_episode.size() << std::endl;
            while (current_episode_it != current_episode.end()) {
                State_Action sa = current_episode_it->first;
                int state = sa.first;
                double R = current_episode_it->second;
                current_episode_it++;
                if (current_episode_it == current_episode.end())break;
                int next_state = current_episode_it->first.first;
                V[state] += alpha * (R + gamma * V[next_state] - V[state]);
//                std::cout << "Episode " << episode_n << ": v[" << state << "]" << Vep[state] << std::endl;

            }
            episode_it++;
        }
        for (int i = 0; i < n_states; i++) {
            V[i] /= episode_n;
        }
        std::cout << n_actions << " States" << std::endl;
        std::cout << n_actions << " Actions" << std::endl;
        std::cout << num_episodes - 1 << " Episodes" << std::endl;
        std::cout << "Gamma: " << gamma << std::endl;
        std::cout << "Alpha: " << alpha << std::endl;
        for (unsigned int i = 0; i < n_actions; i++) {
            V[inv] /= num_episodes;
            std::cout << "v[s_" << i << "]: " << V[i] << std::endl;
        }
    } else if (method == LP) {
        try {
            int n_actions = -1;
            int n_states = -1;
            GRBEnv env = GRBEnv();
            GRBModel model = GRBModel(env);

            GRBVar *v;
            io::CSVReader<5> in(file);
            in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idstateto", "probability", "reward");
//            idstatefrom,idaction,idstateto,probability,reward
            int from, action, to;
            double prob, rew;
            std::map<std::tuple<int, int, int>, double> P;
            std::map<std::tuple<int, int, int>, double> R;
            std::map<int, std::map<int, std::vector<int>>> s_a;
            std::cout << "Probabilities" << std::endl;
            while (in.read_row(from, action, to, prob, rew)) {
                P.insert(std::make_pair(std::make_tuple(from, action, to), prob));
                R.insert(std::make_pair(std::make_tuple(from, action, to), rew));
                if (from > n_states) {
                    n_states = from;
                }
                if (to > n_states) {
                    n_states = to;
                }
                if (s_a.find(from) == s_a.end()) {
                    std::map<int, std::vector<int>> vec;
                    vec.insert(std::make_pair(action, std::vector<int>()));
                    s_a.insert(std::make_pair(from, vec));
                }
                if (s_a.at(from).find(action) == s_a.at(from).end()) {
                    s_a.at(from).insert(std::make_pair(action, std::vector<int>()));
                }
                s_a.at(from).at(action).emplace_back(to);
                if (action > n_actions) {
                    n_actions = action;
                }
            }
            n_states++;
            GRBLinExpr obj = 0.0;
            v = model.addVars(n_states, GRB_CONTINUOUS);
            std::cout << "Variables" << std::endl;
            for (unsigned int i = 0; i < n_states; i++) {
                v[i].set(GRB_DoubleAttr_Obj, 0.0);
                v[i].set(GRB_StringAttr_VarName, std::to_string(i));
                obj += v[i];
            }
            std::cout << "Set Objective" << std::endl;
            model.setObjective(obj, GRB_MINIMIZE);
            std::cout << "Constraints" << std::endl;

            std::map<int, std::map<int, std::vector<int>>>::iterator sa_iter;
            for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                int s = sa_iter->first;
                std::map<int, std::vector<int>> vec = sa_iter->second;
                std::map<int, std::vector<int>>::iterator iter2;
                for (iter2 = vec.begin(); iter2 != vec.end(); iter2++) {
                    GRBLinExpr sum_p;
                    GRBLinExpr sum_r;
                    int ac = iter2->first;
                    std::vector<int> dests = iter2->second;
                    std::cout << "------------" << std::endl;
                    for (auto dest : dests) {
                        std::cout << s << " *" << ac << " " << dest << " "
                                  << P.at(std::make_tuple(s, ac, dest)) << " "
                                  << R.at(std::make_tuple(s, ac, dest)) << std::endl;
                        sum_p += gamma * P.at(std::make_tuple(s, ac, dest)) * v[dest];
                        sum_r += P.at(std::make_tuple(s, ac, dest)) *
                                 R.at(std::make_tuple(s, ac, dest));
                    }
                    model.addConstr(v[s] - sum_p >= sum_r,
                                    "c" + std::to_string(s) + "_" + std::to_string(ac));
                }
            }
            std::cout << "Optimizing" << std::endl;
            model.optimize();
            std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
            int policy[n_states] = {0};
            for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                int s = sa_iter->first;
                double max_val = 0.0;
                int max_ac = 0;
                std::map<int, std::vector<int>> vec = sa_iter->second;
                std::map<int, std::vector<int>>::iterator iter2;
                for (iter2 = vec.begin(); iter2 != vec.end(); iter2++) {
                    int ac = iter2->first;
                    std::vector<int> dests = iter2->second;
                    double val = 0.0;
                    for (auto dest : dests) {
                        val += gamma * P.at(std::make_tuple(s, ac, dest)) * v[dest].get(GRB_DoubleAttr_X) +
                               P.at(std::make_tuple(s, ac, dest)) *
                               R.at(std::make_tuple(s, ac, dest));
                    }
                    if (val > max_val) {
                        max_val = val;
                        max_ac = ac;
                    }
                }
                policy[s] = max_ac;
            }
            std::cout << "Solution: " << std::endl;
            for (int i = 0; i < n_states; i++) {
                std::cout << v[i].get(GRB_StringAttr_VarName) << ": " << v[i].get(GRB_DoubleAttr_X) << std::endl;
            }
            for (int i = 0; i < n_states; i++) {
                std::cout << "π[" << i << "] :" << policy[i] << std::endl;
            }
            delete v;
        } catch (GRBException e) {
            std::cout << "Error code = " << e.getErrorCode() << std::endl;
            std::cout << e.getMessage() << std::endl;
        } catch (...) {
            std::cout << "Exception during optimization" << std::endl;
        }
    } else if (method == VI) {
        io::CSVReader<5> in(file);
        in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idstateto", "probability", "reward");
//            idstatefrom,idaction,idstateto,probability,reward
        int from, action, to, n_states = 0, n_actions = 0;
        double prob, rew;
        std::map<std::tuple<int, int, int>, double> P;
        std::map<std::tuple<int, int, int>, double> R;
        std::map<int, std::map<int, std::vector<int>>> s_a;
        std::cout << "Probabilities" << std::endl;
        while (in.read_row(from, action, to, prob, rew)) {
            P.insert(std::make_pair(std::make_tuple(from, action, to), prob));
            R.insert(std::make_pair(std::make_tuple(from, action, to), rew));
            if (from > n_states) {
                n_states = from;
            }
            if (to > n_states) {
                n_states = to;
            }
            if (s_a.find(from) == s_a.end()) {
                std::map<int, std::vector<int>> vec;
                vec.insert(std::make_pair(action, std::vector<int>()));
                s_a.insert(std::make_pair(from, vec));
            }
            if (s_a.at(from).find(action) == s_a.at(from).end()) {
                s_a.at(from).insert(std::make_pair(action, std::vector<int>()));
            }
            s_a.at(from).at(action).emplace_back(to);
            if (action > n_actions) {
                n_actions = action;
            }
        }
        double v[n_states] = {0};
        int policy[n_states] = {0};
        std::map<int, std::map<int, std::vector<int>>>::iterator sa_iter;

        while (true) {
            double delta = 0.0;
            for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                int s = sa_iter->first;
                double max_val = 0.0;
                int max_ac = 0;
                std::map<int, std::vector<int>> vec = sa_iter->second;
                std::map<int, std::vector<int>>::iterator iter2;
                double temp = v[s];
                for (iter2 = vec.begin(); iter2 != vec.end(); iter2++) {
                    int ac = iter2->first;
                    std::vector<int> dests = iter2->second;
                    double val = 0.0;
                    for (auto dest : dests) {
                        val += P.at(std::make_tuple(s, ac, dest)) * (gamma * v[dest] +
                                                                     R.at(std::make_tuple(s, ac, dest)));
                    }
                    if (val > max_val) {
                        max_val = val;
                        max_ac = ac;
                    }
                }
                policy[s] = max_ac;
                v[s] = max_val;
                delta = std::max(delta, std::abs(temp - v[s]));
            }
            std::cout << "Delta :" << delta << std::endl;
            if (delta < threshold) {
                break;
            }
        }
        for (int i = 0; i < n_states; i++) {
            std::cout << "v[" << i << "]:" << v[i] << std::endl;
        }
        for (int i = 0; i < n_states; i++) {
            std::cout << "π[" << i << "]:" << policy[i] << std::endl;
        }
    } else if (method == PI) {

        io::CSVReader<5> in(file);
        in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idstateto", "probability", "reward");
//            idstatefrom,idaction,idstateto,probability,reward
        int from, action, to, n_states = 0, n_actions = 0;
        double prob, rew;
        std::map<std::tuple<int, int, int>, double> P;
        std::map<std::tuple<int, int, int>, double> R;
        std::map<int, std::map<int, std::vector<int>>> s_a;
        std::cout << "Probabilities" << std::endl;
        while (in.read_row(from, action, to, prob, rew)) {
            P.insert(std::make_pair(std::make_tuple(from, action, to), prob));
            R.insert(std::make_pair(std::make_tuple(from, action, to), rew));
            if (from > n_states) {
                n_states = from;
            }
            if (to > n_states) {
                n_states = to;
            }
            if (s_a.find(from) == s_a.end()) {
                std::map<int, std::vector<int>> vec;
                vec.insert(std::make_pair(action, std::vector<int>()));
                s_a.insert(std::make_pair(from, vec));
            }
            if (s_a.at(from).find(action) == s_a.at(from).end()) {
                s_a.at(from).insert(std::make_pair(action, std::vector<int>()));
            }
            s_a.at(from).at(action).emplace_back(to);
            if (action > n_actions) {
                n_actions = action;
            }
        }
        double v[n_states] = {0};
        int policy[n_states] = {0};
        std::map<int, std::map<int, std::vector<int>>>::iterator sa_iter;
        std::map<int, std::vector<int>>::iterator iter2;

        for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
            int s = sa_iter->first;
            policy[s] = sa_iter->second.begin()->first;
        }
        while (true) {
            // POLICY EVALUATION
            while (true) {
                double delta = 0;
                for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                    int s = sa_iter->first;
                    int ac = policy[s];
                    double val = 0.0;
                    double temp = v[s];
                    std::vector<int> vec = sa_iter->second.find(ac)->second;
                    std::vector<int>::iterator vec_it;
                    for (vec_it = vec.begin(); vec_it != vec.end(); vec_it++) {
                        int dest = *vec_it;
                        val += P.at(std::make_tuple(s, ac, dest)) *
                               (R.at(std::make_tuple(s, ac, dest)) + gamma * v[dest]);
                        v[s] = val;

                    }
                    delta = std::max(delta, std::abs(temp - v[s]));
                }
                std::cout << "delta: " << delta << std::endl;
                if (delta < threshold) {
                    break;
                }
            }
            //POLICY IMPROVEMENT
            bool policy_stable = true;
            for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                int s = sa_iter->first;
                double max_val = 0.0;
                int max_ac = 0;
                std::map<int, std::vector<int>> vec = sa_iter->second;
                std::map<int, std::vector<int>>::iterator iter2;
                double temp = policy[s];
                for (iter2 = vec.begin(); iter2 != vec.end(); iter2++) {
                    int ac = iter2->first;
                    std::vector<int> dests = iter2->second;
                    double val = 0.0;
                    for (auto dest : dests) {
                        val += P.at(std::make_tuple(s, ac, dest)) * (gamma * v[dest] +
                                                                     R.at(std::make_tuple(s, ac, dest)));
                    }
                    if (val > max_val) {
                        max_val = val;
                        max_ac = ac;
                    }
                }
                policy[s] = max_ac;
                if (policy[s] != temp) policy_stable = false;
            }
            if (policy_stable) break;
        }
        for (int i = 0; i < n_states; i++) {
            std::cout << "v[" << i << "]:" << v[i] << std::endl;
        }
        for (int i = 0; i < n_states; i++) {
            std::cout << "π[" << i << "]:" << policy[i] << std::endl;
        }
    } else if (method == MPI) {

        io::CSVReader<5> in(file);
        in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idstateto", "probability", "reward");
//            idstatefrom,idaction,idstateto,probability,reward
        int from, action, to, n_states = 0, n_actions = 0;
        double prob, rew;
        std::map<std::tuple<int, int, int>, double> P;
        std::map<std::tuple<int, int, int>, double> R;
        std::map<int, std::map<int, std::vector<int>>> s_a;
        std::cout << "Probabilities" << std::endl;
        while (in.read_row(from, action, to, prob, rew)) {
            P.insert(std::make_pair(std::make_tuple(from, action, to), prob));
            R.insert(std::make_pair(std::make_tuple(from, action, to), rew));
            if (from > n_states) {
                n_states = from;
            }
            if (to > n_states) {
                n_states = to;
            }
            if (s_a.find(from) == s_a.end()) {
                std::map<int, std::vector<int>> vec;
                vec.insert(std::make_pair(action, std::vector<int>()));
                s_a.insert(std::make_pair(from, vec));
            }
            if (s_a.at(from).find(action) == s_a.at(from).end()) {
                s_a.at(from).insert(std::make_pair(action, std::vector<int>()));
            }
            s_a.at(from).at(action).emplace_back(to);
            if (action > n_actions) {
                n_actions = action;
            }
        }
        double v[n_states] = {0};
        int policy[n_states] = {0};
        std::map<int, std::map<int, std::vector<int>>>::iterator sa_iter;

        for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
            int s = sa_iter->first;
            policy[s] = sa_iter->second.begin()->first;
        }
        std::map<int, std::vector<int>>::iterator action_dest_iterator;
        while (true) {
            // POLICY EVALUATION
            int count = 100;
            while (count > 0) {
                count--;
                double delta = 0;
                for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                    int s = sa_iter->first;
                    int ac = policy[s];
                    double val = 0.0;
                    double temp = v[s];
                    std::vector<int> vec = sa_iter->second.find(ac)->second;
                    std::vector<int>::iterator vec_it;
                    for (vec_it = vec.begin(); vec_it != vec.end(); vec_it++) {
                        int dest = *vec_it;
                        val += P.at(std::make_tuple(s, ac, dest)) *
                               (R.at(std::make_tuple(s, ac, dest)) + gamma * v[dest]);
                        v[s] = val;

                    }
                    delta = std::max(delta, std::abs(temp - v[s]));
                }
                std::cout << "delta: " << delta << std::endl;
                if (delta < threshold) {
                    break;
                }
            }
            //POLICY IMPROVEMENT
            bool policy_stable = true;
            for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                int s = sa_iter->first;
                double max_val = 0.0;
                int max_ac = 0;
                std::map<int, std::vector<int>> vec = sa_iter->second;

                double temp = policy[s];
                for (action_dest_iterator = vec.begin(); action_dest_iterator != vec.end(); action_dest_iterator++) {
                    int ac = action_dest_iterator->first;
                    std::vector<int> dests = action_dest_iterator->second;
                    double val = 0.0;
                    for (auto dest : dests) {
                        val += P.at(std::make_tuple(s, ac, dest)) * (gamma * v[dest] +
                                                                     R.at(std::make_tuple(s, ac, dest)));
                    }
                    if (val > max_val) {
                        max_val = val;
                        max_ac = ac;
                    }
                }
                policy[s] = max_ac;
                if (policy[s] != temp) policy_stable = false;
            }
            if (policy_stable) break;
        }
        for (int i = 0; i < n_states; i++) {
            std::cout << "v[" << i << "]:" << v[i] << std::endl;
        }
        for (int i = 0; i < n_states; i++) {
            std::cout << "π[" << i << "]:" << policy[i] << std::endl;
        }
    }
    return EXIT_SUCCESS;
}