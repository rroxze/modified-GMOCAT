import sys
import json
import os
import math
import yaml
import torch
import dgl
import random
import copy as cp
import numpy as np
import logging
from util import *
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from .dataset import TrainDataset
from .Env import Env

class GCATEnv(Env):
    def __init__(self, args):
        super(GCATEnv, self).__init__(args)
        
        self.nov_reward_map = self.load_nov_reward()
        self.concept_importance = self.load_concept_importance()
        
        if args.target_concepts != [0]:
          self.target_concepts = set(args.target_concepts)
        else:
          self.target_concepts = set(concept for concepts in self.know_map.values() for concept in concepts)

    def load_concept_importance(self):
        path = f'graph_data/{self.args.data_name}/K_Directed.txt'
        degrees = defaultdict(int)
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        degrees[u] += 1
                        degrees[v] += 1
        except FileNotFoundError:
            # If no graph file, assume uniform importance
            return defaultdict(lambda: 1.0)

        # Normalize
        max_deg = max(degrees.values()) if degrees else 1
        importance = {k: v / max_deg for k, v in degrees.items()}
        return importance

    def reset_with_users(self, uids):
        self.state = {}
        self.avail_questions = {}
        self.used_questions = {}
        self.know_stat = {}
        self.concept_consistency = {}

        for uu in uids:
            self.avail_questions[uu] = {
                qid for qid in self.sup_rates[uu].keys() if any(concept in self.target_concepts for concept in self.know_map[qid])
            }
            self.used_questions[uu] = []
            self.concept_consistency[uu] = {concept: False for concept in self.target_concepts}
        
        self.uids = uids
        self.cnt_step = 0
        self.state['batch_question']= np.zeros((len(uids), len(self.target_concepts)*2))
        self.state['batch_answer']= np.zeros((len(uids), len(self.target_concepts)*2))
        self.last_div = 0

        # Pass concept_map to get uncertainty if available
        result = self.model.get_knowledge_status(torch.tensor(self.uids).to(self.device), self.know_map)

        if isinstance(result, tuple):
            self.know_stat, self.uncertainty = result
        else:
            self.know_stat = result
            self.uncertainty = None

        self.previous_know_stat = self.know_stat * 0
        
        _, pred, correct_query = self.model.cal_loss(self.uids, self.query_rates, self.know_map)
        self.last_accuracy = np.zeros(len(uids))
        for i in range(len(pred)):
            pred_bin = np.where(pred[i] > 0.5, 1, 0)
            ACC = np.sum(np.equal(pred_bin, correct_query[i])) / len(pred_bin) 
            self.last_accuracy[i] = ACC

        return self.state

    def step(self, action, last_epoch):
        for i, uu in enumerate(self.uids):
            if action[i] not in self.avail_questions[uu]:
                self.logger.info("action exited")
                pass

        reward, pred, label, rate = self.reward(action)
        
        self.previous_know_stat = self.know_stat

        result = self.model.get_knowledge_status(torch.tensor(self.uids).to(self.device), self.know_map)
        if isinstance(result, tuple):
            self.know_stat, self.uncertainty = result
        else:
            self.know_stat = result
            self.uncertainty = None
        
        done = False

        coverages = []
        for i, uu in enumerate(self.uids):
                    
            self.avail_questions[uu].remove(action[i])
            self.used_questions[uu].append(action[i])
            
            for concept in self.know_map[action[i]]:
                if concept in self.target_concepts:

                  is_stable = False

                  if self.uncertainty is not None:
                      # Uncertainty-Based Termination
                      # self.uncertainty is (batch_size, num_concepts) variance
                      unc_val = self.uncertainty[i][concept]
                      # Threshold for variance. If variance is low, we are certain.
                      # Max variance of a probability p is 0.25 (at p=0.5).
                      # Let's say threshold is 0.01 (std dev 0.1).
                      if unc_val < 0.01:
                          is_stable = True
                          # print(f"Concept {concept} stable by uncertainty: {unc_val.item()}")
                  else:
                      # Fallback to delta
                      prev_score = self.previous_know_stat[i][concept]
                      curr_score = self.know_stat[i][concept]
                      error = abs(curr_score - prev_score)
                      if error < 0.010:
                          is_stable = True

                  if is_stable:
                      self.concept_consistency[uu][concept] = True
                      stable_questions = []
                      for qid in self.avail_questions[uu]:
                          concepts_in_question = self.know_map[qid]
                          all_concepts_stable = all(
                              concept in self.concept_consistency[uu] and self.concept_consistency[uu][concept]
                              for concept in concepts_in_question
                          )
                          
                          if all_concepts_stable:
                              stable_questions.append(qid)

                      for qid in stable_questions:
                          self.avail_questions[uu].remove(qid)

            all_concepts = set()
            tested_concepts = set()
            for qid in self.rates[uu]: 
                all_concepts.update(set(self.know_map[qid]))
            for qid in self.used_questions[uu]:
                tested_concepts.update(set(self.know_map[qid]))

            if len(all_concepts) > 0:
                coverage = len(tested_concepts) / len(all_concepts)
            else:
                coverage = 0.0
            coverages.append(coverage)

            all_stable = all(
                self.concept_consistency[uu][concept] 
                for concept in self.target_concepts
            )

            if (coverage == 1.0 and all_stable) or len(self.avail_questions[uu])<1:
              done = True
              
        cov = sum(coverages) / len(coverages)

        all_info = [{"pred": pred[i], "label": label[i], "rate":rate[i]} for i, uu in enumerate(self.uids)]

        self.cnt_step += 1

        self.state['batch_question'][:, self.cnt_step] = action
        self.state['batch_answer'][:, self.cnt_step] = np.array(rate)+1 # idx=0 is pad

        return self.state, reward, done, all_info, cov

    def reward(self, action):
        # update cdm
        records = [(uu, action[i], self.rates[uu][action[i]]) for i, uu in enumerate(self.uids)]
        self.dataset = TrainDataset(records, self.know_map, self.user_num, self.item_num, self.know_num)
        self.model.update(self.dataset, self.args.cdm_lr, epochs=self.args.cdm_epoch, batch_size=self.args.cdm_bs)

        # eval on query
        loss, pred, correct_query = self.model.cal_loss(self.uids, self.query_rates, self.know_map)
        final_rate = [self.rates[uu][action[i]] for i, uu in enumerate(self.uids)]

        new_accuracy = np.zeros(len(self.uids))
        for i in range(len(pred)):
            pred_bin = np.where(pred[i] > 0.5, 1, 0)
            ACC = np.sum(np.equal(pred_bin, correct_query[i])) / len(pred_bin) 
            new_accuracy[i] = ACC

        acc_rwd = new_accuracy - self.last_accuracy
        self.last_accuracy = new_accuracy

        div_rwd = []
        for i, uu in enumerate(self.uids):
             # Calculate current coverage for user uu
             all_concepts = set()
             tested_concepts = set()
             for qid in self.rates[uu]:
                 all_concepts.update(set(self.know_map[qid]))
             for qid in self.used_questions[uu]:
                 tested_concepts.update(set(self.know_map[qid]))

             if len(all_concepts) > 0:
                 cov = len(tested_concepts) / len(all_concepts)
             else:
                 cov = 0.0

             r = self.compute_div_reward(list(self.sup_rates[uu].keys()), self.know_map, self.used_questions[uu], action[i], self.concept_consistency[uu], cov)
             div_rwd.append(r)
        div_rwd = np.array(div_rwd)

        nov_rwd = np.array([self.nov_reward_map[action[i]] for i in range(len(action))])

        rwd = np.concatenate((acc_rwd.reshape(-1,1), div_rwd.reshape(-1,1), nov_rwd.reshape(-1,1)),axis=-1) # (B,3)
        
        return rwd, pred, correct_query, final_rate
    
    def load_nov_reward(self):
        with open(f'data/nov_reward_{self.args.data_name}.json', encoding='utf8') as i_f:
            concept_data = json.load(i_f) 
        
        nov_reward_map = {}
        for k,v in concept_data.items():
            qid_pad = int(k) +1
            nov_reward_map[qid_pad] = v
            
        return nov_reward_map
    
    def compute_div_reward(self, all_questions, concept_map, tested_questions, qid, concept_consistency, coverage=0.0):
        concept_cnt = set()
        
        for q in list(tested_questions):
            for c in concept_map[q]:
                concept_cnt.add(c)
        
        reward = 0.0

        # Calculate coverage deficit
        deficit = 1.0 - coverage

        for c in concept_map[qid]:
            imp = self.concept_importance.get(c, 0.5) # Default 0.5
            if c not in concept_cnt:
                # New concept
                reward += imp
            elif not concept_consistency.get(c, False):
                # Unstable concept
                reward += imp

        # Scale by deficit (boost reward if coverage is low)
        return reward * (1.0 + deficit)
