# !/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        if model_name == 'TransQuatE':
            self.entity_dim = hidden_dim * 4
            self.relation_dim = hidden_dim * 4

        # TODO: initalization change later
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        if model_name == 'TransQuatE':
            self.rotator_head_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.rotator_head_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            # TODO: relation tane mi head tane mi

            self.mapping_regularizer_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.mapping_regularizer_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'TransQuatE', 'TransQuatE0']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        # if model_name == 'TransQuatE' and (not double_entity_embedding or not double_relation_embedding):
        #    raise ValueError('TransQuatE should use --double_entity_embedding and --double_relation_embedding')


    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            # print("batch_size", batch_size)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]  # head
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]  # relation
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]  # tail
            ).unsqueeze(1)

            if self.model_name == 'TransQuatE':
                # this part has same dimensions as head
                mapping_regularizer = torch.index_select(
                    self.mapping_regularizer_embedding,
                    dim=0,
                    index=sample[:, 0]  # head
                ).unsqueeze(1)

                # TODO: check
                rotator_head = torch.index_select(
                    self.rotator_head_embedding,
                    dim=0,
                    index= sample[:, 1]  # relation
                ).unsqueeze(1)

            #TODO: select rotators here

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            # this forward is used during the evaluation too
            #print('head-batch')
            #print("tail_part_first", tail_part.size(0))
            #print("tail_part_second", tail_part.size(1))
            #print("batch_size = head_part_first", head_part.size(0))
            #print("negative_sample_size = head_part_second", head_part.size(1))

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1) # batch size kadar negative headler yada all entityler

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]  # batch size kadar relation
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2] # batch size kadar tail
            ).unsqueeze(1)

            if self.model_name == 'TransQuatE':
                # this part has same dimensions as head
                mapping_regularizer = torch.index_select(
                    self.mapping_regularizer_embedding,
                    dim=0,
                    index=head_part.view(-1)  # negative headler için
                ).view(batch_size, negative_sample_size, -1) # batch size kadar negative headler yada all entityler

                # TODO: check
                rotator_head = torch.index_select(
                    self.rotator_head_embedding,
                    dim=0,
                    index=tail_part[:, 1]  # relation
                ).unsqueeze(1) # relation tane head rotator

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            #print('tail-batch')
            #print("head_part_first", head_part.size(0))
            #print("head_part_second", head_part.size(1))
            #print("batch_size = tail_part_first", tail_part.size(0))
            #print("negative_sample_size = tail_part_second", tail_part.size(1))

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]  # batch size kadar head
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]  # batch size kadar relation
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1) # batch size kadar negative tailler yada all entityler

            if self.model_name == 'TransQuatE':
                # this part has same dimensions as head
                mapping_regularizer = torch.index_select(
                    self.mapping_regularizer_embedding,
                    dim=0,
                    index=head_part[:, 0]  # head
                ).unsqueeze(1)

                # TODO: check
                rotator_head = torch.index_select(
                    self.rotator_head_embedding,
                    dim=0,
                    index=head_part[:, 1]  # relation
                ).unsqueeze(1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'TransQuatE': self.TransQuatE,
            'TransQuatE0': self.TransQuatE0
        }

        if self.model_name == 'TransQuatE':
            score = model_func[self.model_name](head, relation, tail, rotator_head,
                                                mapping_regularizer, mode)
        elif self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):

        # dimension check

        #print("mode", mode)
        #print("head", head.shape)
        #print("relation", relation.shape)
        #print("tail", tail.shape)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TransQuatE(self, head, relation, tail, rotator_head, mapping_regularizer, mode):

        # dimension check

        #print("mode", mode)
        #print("head", head.shape)
        #print("relation", relation.shape)
        #print("tail", tail.shape)
        #print("rotator_head", rotator_head.shape)
        #print("mapping_regularizer", mapping_regularizer.shape)


        # TODO: head-batch taiil-batch difference?
        head, head_i, head_j, head_k = torch.chunk(head, 4, dim=2)
        rot_h, rot_hi, rot_hj, rot_hk = self.normalize_quaternion(rotator_head)  # relationdan gelen bir rotation degree

        # rotating head in 4 dim --- checked
        rotated_head_real = head * rot_h - head_i * rot_hi - head_j * rot_hj - head_k * rot_hk
        rotated_head_i = head * rot_hi + head_i * rot_h + head_j * rot_hk - head_k * rot_hj
        rotated_head_j = head * rot_hj - head_i * rot_hk + head_j * rot_h + head_k * rot_hi
        rotated_head_k = head * rot_hk + head_i * rot_hj - head_j * rot_hi + head_k * rot_h

        # now translating head
        tran_real, tran_i, tran_j, tran_k = torch.chunk(relation, 4, dim=2)  # by relation
        translated_head_real = rotated_head_real + tran_real
        translated_head_i = rotated_head_i + tran_i
        translated_head_j = rotated_head_j + tran_j
        translated_head_k = rotated_head_k + tran_k
        # -----------------------------------------------------#
        #mapping_regularizer_real, mapping_regularizer_i, mapping_regularizer_j, mapping_regularizer_k = \
        #    torch.chunk(mapping_regularizer, 4, dim=2)
        # TODO:
        mapping_regularizer_real, mapping_regularizer_i, mapping_regularizer_j, mapping_regularizer_k = self.normalize_quaternion(mapping_regularizer)

        tail_real, tail_i, tail_j, tail_k = torch.chunk(tail, 4, dim=2)
        # TODO:Ask
        # tail_i, tail_j, tail_k = -tail_i, -tail_j, -tail_k

        # rotating mapping_regularizer in 4 dim by tail
        # --- checked
        rotated_mapping_regularizer_real = mapping_regularizer_real * tail_real - mapping_regularizer_i * tail_i - mapping_regularizer_j * tail_j - mapping_regularizer_k * tail_k
        rotated_mapping_regularizer_i = mapping_regularizer_real * tail_i + mapping_regularizer_i * tail_real + mapping_regularizer_j * tail_k - mapping_regularizer_k * tail_j
        rotated_mapping_regularizer_j = mapping_regularizer_real * tail_j - mapping_regularizer_i * tail_k + mapping_regularizer_j * tail_real + mapping_regularizer_k * tail_i
        rotated_mapping_regularizer_k = mapping_regularizer_real * tail_k + mapping_regularizer_i * tail_j - mapping_regularizer_j * tail_i + mapping_regularizer_k * tail_real

        score_r = translated_head_real - rotated_mapping_regularizer_real
        score_i = translated_head_i - rotated_mapping_regularizer_i
        score_j = translated_head_j - rotated_mapping_regularizer_j
        score_k = translated_head_k - rotated_mapping_regularizer_k

        score = torch.stack([score_r, score_i, score_j, score_k], dim=0)
        score = score.norm(dim=0)
        score = score.sum(dim=2)

        # if mode == 'head-batch':
        #    score = head + (relation - tail)
        # else:
        #    score = (head + relation) - tail

        return score

    def TransQuatE0(self, head, relation, tail, rotator_head, mapping_regularizer, mode):

        # dimension check

        #print("mode", mode)
        #print("head", head.shape)
        #print("relation", relation.shape)
        #print("tail", tail.shape)
        #print("rotator_head", rotator_head.shape)
        #print("mapping_regularizer", mapping_regularizer.shape)


        # TODO: head-batch taiil-batch difference?
        head, head_i, head_j, head_k = torch.chunk(head, 4, dim=2)
        rot_h, rot_hi, rot_hj, rot_hk = self.normalize_quaternion(rotator_head)  # relationdan gelen bir rotation degree

        # rotating head in 4 dim --- checked
        rotated_head_real = head * rot_h - head_i * rot_hi - head_j * rot_hj - head_k * rot_hk
        rotated_head_i = head * rot_hi + head_i * rot_h + head_j * rot_hk - head_k * rot_hj
        rotated_head_j = head * rot_hj - head_i * rot_hk + head_j * rot_h + head_k * rot_hi
        rotated_head_k = head * rot_hk + head_i * rot_hj - head_j * rot_hi + head_k * rot_h

        # now translating head
        tran_real, tran_i, tran_j, tran_k = torch.chunk(relation, 4, dim=2)  # by relation
        translated_head_real = rotated_head_real + tran_real
        translated_head_i = rotated_head_i + tran_i
        translated_head_j = rotated_head_j + tran_j
        translated_head_k = rotated_head_k + tran_k
        # -----------------------------------------------------#
        mapping_regularizer_real, mapping_regularizer_i, mapping_regularizer_j, mapping_regularizer_k = \
            torch.chunk(mapping_regularizer, 4, dim=2)
        # TODO:
        # mapping_regularizer_real, mapping_regularizer_i, mapping_regularizer_j, mapping_regularizer_k = self.normalize_quaternion(mapping_regularizer)

        tail_real, tail_i, tail_j, tail_k = torch.chunk(tail, 4, dim=2)
        # TODO:Ask
        # tail_i, tail_j, tail_k = -tail_i, -tail_j, -tail_k

        # rotating mapping_regularizer in 4 dim by tail
        # --- checked
        rotated_mapping_regularizer_real = mapping_regularizer_real * tail_real - mapping_regularizer_i * tail_i - mapping_regularizer_j * tail_j - mapping_regularizer_k * tail_k
        rotated_mapping_regularizer_i = mapping_regularizer_real * tail_i + mapping_regularizer_i * tail_real + mapping_regularizer_j * tail_k - mapping_regularizer_k * tail_j
        rotated_mapping_regularizer_j = mapping_regularizer_real * tail_j - mapping_regularizer_i * tail_k + mapping_regularizer_j * tail_real + mapping_regularizer_k * tail_i
        rotated_mapping_regularizer_k = mapping_regularizer_real * tail_k + mapping_regularizer_i * tail_j - mapping_regularizer_j * tail_i + mapping_regularizer_k * tail_real

        score_r = translated_head_real - rotated_mapping_regularizer_real
        score_i = translated_head_i - rotated_mapping_regularizer_i
        score_j = translated_head_j - rotated_mapping_regularizer_j
        score_k = translated_head_k - rotated_mapping_regularizer_k

        score = torch.stack([score_r, score_i, score_j, score_k], dim=0)
        score = score.norm(dim=0)
        score = score.sum(dim=2)

        # if mode == 'head-batch':
        #    score = head + (relation - tail)
        # else:
        #    score = (head + relation) - tail

        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    #utility
    def normalize_quaternion(self, tensor):
        t, ti, tj, tk = torch.chunk(tensor, 4, dim=2)
        denom = torch.sqrt(t ** 2 + ti ** 2 + tj ** 2 + tk ** 2)
        t = t / denom
        ti = ti / denom
        tj = tj / denom
        tk = tk / denom
        return t, ti, tj, tk

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # print("positive_sample", positive_sample.shape)  positive_sample torch.Size([512, 3])
        # print("negative_sample", negative_sample.shape)  negative_sample torch.Size([512, 10])
        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        # print(loss)
        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            ''' 
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )
            '''
            # SD2020 de elimde tail corrupted triplelar olmalı test için sadece
            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            #test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            test_dataset_list = [test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics

    #TODO: here is the original part
    @staticmethod
    def test_step_original(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
