/*
 * @file main.cpp
 *
 * @author Andreas Lydakis andlydakis@gmail.com
 * @date 9/26/2017
 * @brief MDP Solver that uses a variety of algorithm
 *
 */
#include <iostream>
#include <fstream>
#include <set>
#include <cmath>
#include <vector>
#include "csv.h"
#include "/usr/local/lib/cxxopts/include/cxxopts.hpp"
#include "gurobi_c++.h"

enum class Method_t {
    LP, VI, PI, MPI, TD0, EVMC
};

using namespace std;

int main(int argc, char *argv[]) {
    cxxopts::Options options(argv[0], " - example command line options");
    options.add_options()
//            ("h, help", "Display help message.")
            ("f,file", "File name", cxxopts::value<string>()->default_value("./samples.csv"), "")
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
            ("m, method", "Algorithm to use", cxxopts::value<string>()
                    ->default_value("TD0"), "M");
    try {
        options.parse(argc, argv);
    } catch (cxxopts::OptionException oe) {
        cout << oe.what() << endl << endl << " *** usage *** " << endl;
        cout << options.help() << endl;
        return -1;
    }

    cout << "Parsed Args" << endl;
    string file = options["file"].as<string>();
    double gamma = options["gamma"].as<double>();
    double alpha = options["alpha"].as<double>();
    double threshold = options["threshold"].as<double>();
    int iter = options["iterations"].as<int>();
    string m = options["method"].as<string>();

    Method_t method;
    if (m == "LP") {
        method = Method_t::LP;
    } else if (m == "VI") {
        method = Method_t::VI;
    } else if (m == "PI") {
        method = Method_t::PI;
    } else if (m == "MPI") {
        method = Method_t::MPI;
    } else if (m == "TD0") {
        method = Method_t::TD0;
    } else if (m == "EVMC") {
        method = Method_t::EVMC;
    } else {
        cout << "Unknown optimization method " << m << ". Terminating." << endl;
        terminate();
    }

    cout << "Reading: " << file << endl;

    if (method == Method_t::TD0) {
        vector<double> valuef(0, 0.0);
        // state and action variables
        int step, inv, ord;
        int cur_inv, cur_ord;
        // rewards
        double rew, cur_rew;

        io::CSVReader<4> in(file);
        in.read_header(io::ignore_extra_column, "Step", "Inventory", "Order", "Reward");

        // iteration number
        int iter = 1;

        // read initial input
        in.read_row(step, cur_inv, cur_ord, cur_rew);

        // parse the input file
        while (in.read_row(step, inv, ord, rew)) {
            // do not update the last step of an episode
            if (step != 1) {
                //cout << step << " " << inv << " " << ord << " " << rew << " " << sim << " " << endl;
                int maxstate = max(cur_inv, inv);
                if (size_t(maxstate) >= valuef.size()) {
                    valuef.resize(maxstate + 1, 0.0);
                }
                valuef[cur_inv] += (alpha / sqrt(iter)) * (cur_rew + gamma * valuef[inv] - valuef[cur_inv]);
                iter++;
            }
            cur_inv = inv;
            cur_rew = rew;
            cur_ord = ord;
        }

        // print statistics
        cout << valuef.size() << " States" << endl;

    } else if (method == Method_t::EVMC) {
        int n_actions = 0;

        int step, inv, ord;
        int num_episodes = 0;
        double rew;
        string sim;

        typedef pair<int, int> State_Action;
        typedef pair<State_Action, double> Ep_Step;
        typedef vector<Ep_Step> Episode;
        map<int, double> values;
        vector<Episode> episodes;

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
//            cout << step << " " << sa.first << " " << sa.second << " " << rew << " " << sim << " " << endl;
            if (ord > n_actions) n_actions = ord;
        }
        cout << episodes.size() << " episodes" << endl;
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
            cout << "Episode Size: " << current_episode.size() << endl;
            while (current_episode_it != current_episode.end()) {
                State_Action sa = current_episode_it->first;
                int state = sa.first;
                double R = current_episode_it->second;
                current_episode_it++;
                if (current_episode_it == current_episode.end())break;
                int next_state = current_episode_it->first.first;
                V[state] += alpha * (R + gamma * V[next_state] - V[state]);
//                cout << "Episode " << episode_n << ": v[" << state << "]" << Vep[state] << endl;

            }
            episode_it++;
        }
        for (int i = 0; i < n_states; i++) {
            V[i] /= episode_n;
        }
        cout << n_actions << " States" << endl;
        cout << n_actions << " Actions" << endl;
        cout << num_episodes - 1 << " Episodes" << endl;
        cout << "Gamma: " << gamma << endl;
        cout << "Alpha: " << alpha << endl;
        for (unsigned int i = 0; i < n_actions; i++) {
            V[inv] /= num_episodes;
            cout << "v[s_" << i << "]: " << V[i] << endl;
        }
    } else if (method == Method_t::LP) {
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
            map<tuple<int, int, int>, double> P;
            map<tuple<int, int, int>, double> R;
            map<int, map<int, vector<int>>> s_a;
            cout << "Probabilities" << endl;
            while (in.read_row(from, action, to, prob, rew)) {
                P.insert(make_pair(make_tuple(from, action, to), prob));
                R.insert(make_pair(make_tuple(from, action, to), rew));
                if (from > n_states) {
                    n_states = from;
                }
                if (to > n_states) {
                    n_states = to;
                }
                if (s_a.find(from) == s_a.end()) {
                    map<int, vector<int>> vec;
                    vec.insert(make_pair(action, vector<int>()));
                    s_a.insert(make_pair(from, vec));
                }
                if (s_a.at(from).find(action) == s_a.at(from).end()) {
                    s_a.at(from).insert(make_pair(action, vector<int>()));
                }
                s_a.at(from).at(action).emplace_back(to);
                if (action > n_actions) {
                    n_actions = action;
                }
            }
            n_states++;
            GRBLinExpr obj = 0.0;
            v = model.addVars(n_states, GRB_CONTINUOUS);
            cout << "Variables" << endl;
            for (unsigned int i = 0; i < n_states; i++) {
                v[i].set(GRB_DoubleAttr_Obj, 0.0);
                v[i].set(GRB_StringAttr_VarName, to_string(i));
                obj += v[i];
            }
            cout << "Set Objective" << endl;
            model.setObjective(obj, GRB_MINIMIZE);
            cout << "Constraints" << endl;

            map<int, map<int, vector<int>>>::iterator sa_iter;
            for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                int s = sa_iter->first;
                map<int, vector<int>> vec = sa_iter->second;
                map<int, vector<int>>::iterator iter2;
                for (iter2 = vec.begin(); iter2 != vec.end(); iter2++) {
                    GRBLinExpr sum_p;
                    GRBLinExpr sum_r;
                    int ac = iter2->first;
                    vector<int> dests = iter2->second;
                    cout << "------------" << endl;
                    for (auto dest : dests) {
                        cout << s << " *" << ac << " " << dest << " "
                             << P.at(make_tuple(s, ac, dest)) << " "
                             << R.at(make_tuple(s, ac, dest)) << endl;
                        sum_p += gamma * P.at(make_tuple(s, ac, dest)) * v[dest];
                        sum_r += P.at(make_tuple(s, ac, dest)) *
                                 R.at(make_tuple(s, ac, dest));
                    }
                    model.addConstr(v[s] - sum_p >= sum_r,
                                    "c" + to_string(s) + "_" + to_string(ac));
                }
            }
            cout << "Optimizing" << endl;
            model.optimize();
            cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
            int policy[n_states] = {0};
            for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                int s = sa_iter->first;
                double max_val = 0.0;
                int max_ac = 0;
                map<int, vector<int>> vec = sa_iter->second;
                map<int, vector<int>>::iterator iter2;
                for (iter2 = vec.begin(); iter2 != vec.end(); iter2++) {
                    int ac = iter2->first;
                    vector<int> dests = iter2->second;
                    double val = 0.0;
                    for (auto dest : dests) {
                        val += gamma * P.at(make_tuple(s, ac, dest)) * v[dest].get(GRB_DoubleAttr_X) +
                               P.at(make_tuple(s, ac, dest)) *
                               R.at(make_tuple(s, ac, dest));
                    }
                    if (val > max_val) {
                        max_val = val;
                        max_ac = ac;
                    }
                }
                policy[s] = max_ac;
            }
            cout << "Solution: " << endl;
            for (int i = 0; i < n_states; i++) {
                cout << v[i].get(GRB_StringAttr_VarName) << ": " << v[i].get(GRB_DoubleAttr_X) << endl;
            }
            for (int i = 0; i < n_states; i++) {
                cout << "π[" << i << "] :" << policy[i] << endl;
            }
            delete v;
        } catch (GRBException e) {
            cout << "Error code = " << e.getErrorCode() << endl;
            cout << e.getMessage() << endl;
        } catch (...) {
            cout << "Exception during optimization" << endl;
        }
    } else if (method == Method_t::VI) {
        io::CSVReader<5> in(file);
        in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idstateto", "probability", "reward");
//            idstatefrom,idaction,idstateto,probability,reward
        int from, action, to, n_states = 0, n_actions = 0;
        double prob, rew;
        map<tuple<int, int, int>, double> P;
        map<tuple<int, int, int>, double> R;
        map<int, map<int, vector<int>>> s_a;
        cout << "Probabilities" << endl;
        while (in.read_row(from, action, to, prob, rew)) {
            P.insert(make_pair(make_tuple(from, action, to), prob));
            R.insert(make_pair(make_tuple(from, action, to), rew));
            if (from > n_states) {
                n_states = from;
            }
            if (to > n_states) {
                n_states = to;
            }
            if (s_a.find(from) == s_a.end()) {
                map<int, vector<int>> vec;
                vec.insert(make_pair(action, vector<int>()));
                s_a.insert(make_pair(from, vec));
            }
            if (s_a.at(from).find(action) == s_a.at(from).end()) {
                s_a.at(from).insert(make_pair(action, vector<int>()));
            }
            s_a.at(from).at(action).emplace_back(to);
            if (action > n_actions) {
                n_actions = action;
            }
        }
        double v[n_states] = {0};
        int policy[n_states] = {0};
        map<int, map<int, vector<int>>>::iterator sa_iter;

        while (true) {
            double delta = 0.0;
            for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
                int s = sa_iter->first;
                double max_val = 0.0;
                int max_ac = 0;
                map<int, vector<int>> vec = sa_iter->second;
                map<int, vector<int>>::iterator iter2;
                double temp = v[s];
                for (iter2 = vec.begin(); iter2 != vec.end(); iter2++) {
                    int ac = iter2->first;
                    vector<int> dests = iter2->second;
                    double val = 0.0;
                    for (auto dest : dests) {
                        val += P.at(make_tuple(s, ac, dest)) * (gamma * v[dest] +
                                                                R.at(make_tuple(s, ac, dest)));
                    }
                    if (val > max_val) {
                        max_val = val;
                        max_ac = ac;
                    }
                }
                policy[s] = max_ac;
                v[s] = max_val;
                delta = max(delta, abs(temp - v[s]));
            }
            cout << "Delta :" << delta << endl;
            if (delta < threshold) {
                break;
            }
        }
        for (int i = 0; i < n_states; i++) {
            cout << "v[" << i << "]:" << v[i] << endl;
        }
        for (int i = 0; i < n_states; i++) {
            cout << "π[" << i << "]:" << policy[i] << endl;
        }
    } else if (method == Method_t::PI) {

        io::CSVReader<5> in(file);
        in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idstateto", "probability", "reward");
//            idstatefrom,idaction,idstateto,probability,reward
        int from, action, to, n_states = 0, n_actions = 0;
        double prob, rew;
        map<tuple<int, int, int>, double> P;
        map<tuple<int, int, int>, double> R;
        map<int, map<int, vector<int>>> s_a;
        cout << "Probabilities" << endl;
        while (in.read_row(from, action, to, prob, rew)) {
            P.insert(make_pair(make_tuple(from, action, to), prob));
            R.insert(make_pair(make_tuple(from, action, to), rew));
            if (from > n_states) {
                n_states = from;
            }
            if (to > n_states) {
                n_states = to;
            }
            if (s_a.find(from) == s_a.end()) {
                map<int, vector<int>> vec;
                vec.insert(make_pair(action, vector<int>()));
                s_a.insert(make_pair(from, vec));
            }
            if (s_a.at(from).find(action) == s_a.at(from).end()) {
                s_a.at(from).insert(make_pair(action, vector<int>()));
            }
            s_a.at(from).at(action).emplace_back(to);
            if (action > n_actions) {
                n_actions = action;
            }
        }
        double v[n_states] = {0};
        int policy[n_states] = {0};
        map<int, map<int, vector<int>>>::iterator sa_iter;
        map<int, vector<int>>::iterator iter2;

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
                    vector<int> vec = sa_iter->second.find(ac)->second;
                    vector<int>::iterator vec_it;
                    for (vec_it = vec.begin(); vec_it != vec.end(); vec_it++) {
                        int dest = *vec_it;
                        val += P.at(make_tuple(s, ac, dest)) *
                               (R.at(make_tuple(s, ac, dest)) + gamma * v[dest]);
                        v[s] = val;

                    }
                    delta = max(delta, abs(temp - v[s]));
                }
                cout << "delta: " << delta << endl;
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
                map<int, vector<int>> vec = sa_iter->second;
                map<int, vector<int>>::iterator iter2;
                double temp = policy[s];
                for (iter2 = vec.begin(); iter2 != vec.end(); iter2++) {
                    int ac = iter2->first;
                    vector<int> dests = iter2->second;
                    double val = 0.0;
                    for (auto dest : dests) {
                        val += P.at(make_tuple(s, ac, dest)) * (gamma * v[dest] +
                                                                R.at(make_tuple(s, ac, dest)));
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
            cout << "v[" << i << "]:" << v[i] << endl;
        }
        for (int i = 0; i < n_states; i++) {
            cout << "π[" << i << "]:" << policy[i] << endl;
        }
    } else if (method == Method_t::MPI) {

        io::CSVReader<5> in(file);
        in.read_header(io::ignore_extra_column, "idstatefrom", "idaction", "idstateto", "probability", "reward");
//            idstatefrom,idaction,idstateto,probability,reward
        int from, action, to, n_states = 0, n_actions = 0;
        double prob, rew;
        map<tuple<int, int, int>, double> P;
        map<tuple<int, int, int>, double> R;
        map<int, map<int, vector<int>>> s_a;
        cout << "Probabilities" << endl;
        while (in.read_row(from, action, to, prob, rew)) {
            P.insert(make_pair(make_tuple(from, action, to), prob));
            R.insert(make_pair(make_tuple(from, action, to), rew));
            if (from > n_states) {
                n_states = from;
            }
            if (to > n_states) {
                n_states = to;
            }
            if (s_a.find(from) == s_a.end()) {
                map<int, vector<int>> vec;
                vec.insert(make_pair(action, vector<int>()));
                s_a.insert(make_pair(from, vec));
            }
            if (s_a.at(from).find(action) == s_a.at(from).end()) {
                s_a.at(from).insert(make_pair(action, vector<int>()));
            }
            s_a.at(from).at(action).emplace_back(to);
            if (action > n_actions) {
                n_actions = action;
            }
        }
        double v[n_states] = {0};
        int policy[n_states] = {0};
        map<int, map<int, vector<int>>>::iterator sa_iter;

        for (sa_iter = s_a.begin(); sa_iter != s_a.end(); sa_iter++) {
            int s = sa_iter->first;
            policy[s] = sa_iter->second.begin()->first;
        }
        map<int, vector<int>>::iterator action_dest_iterator;
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
                    vector<int> vec = sa_iter->second.find(ac)->second;
                    vector<int>::iterator vec_it;
                    for (vec_it = vec.begin(); vec_it != vec.end(); vec_it++) {
                        int dest = *vec_it;
                        val += P.at(make_tuple(s, ac, dest)) *
                               (R.at(make_tuple(s, ac, dest)) + gamma * v[dest]);
                        v[s] = val;

                    }
                    delta = max(delta, abs(temp - v[s]));
                }
                cout << "delta: " << delta << endl;
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
                map<int, vector<int>> vec = sa_iter->second;

                double temp = policy[s];
                for (action_dest_iterator = vec.begin(); action_dest_iterator != vec.end(); action_dest_iterator++) {
                    int ac = action_dest_iterator->first;
                    vector<int> dests = action_dest_iterator->second;
                    double val = 0.0;
                    for (auto dest : dests) {
                        val += P.at(make_tuple(s, ac, dest)) * (gamma * v[dest] +
                                                                R.at(make_tuple(s, ac, dest)));
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
            cout << "v[" << i << "]:" << v[i] << endl;
        }
        for (int i = 0; i < n_states; i++) {
            cout << "π[" << i << "]:" << policy[i] << endl;
        }
    }
    return EXIT_SUCCESS;
}