extern crate ndarray;
extern crate num_cpus;
extern crate rand;

use self::ndarray::{Array2, Array3, Axis};
use crate::clustering::{Clusterings, WorkingClustering};
use crate::log2cache::Log2Cache;
use crate::loss::*;
use crate::*;
use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_isaac::IsaacRng;
use std::convert::TryFrom;
use std::f64;
use std::slice;
use std::sync::mpsc;
use std::time::{Duration, SystemTime};

// **************************************************************
// Implementation of SALSO for losses based on confusion matrices
// **************************************************************

pub trait CMLossComputer {
    #[allow(unused_variables)]
    fn initialize(&mut self, state: &WorkingClustering, cms: &Array3<CountType>) {}

    fn compute_loss(&self, state: &WorkingClustering, cms: &Array3<CountType>) -> f64;

    fn change_in_loss(
        &self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) -> f64 {
        if from_label_option.is_some() && to_label == from_label_option.unwrap() {
            self.compute_loss(&state, &cms)
        } else {
            let mut state = state.clone();
            let mut cms = cms.clone();
            let n_draws = cms.len_of(Axis(2));
            state.reassign(item_index, to_label);
            let to_index = to_label as usize + 1;
            let from_index = if from_label_option.is_some() {
                from_label_option.unwrap() as usize + 1
            } else {
                0
            };
            for draw_index in 0..n_draws {
                let other_index = draws.label(draw_index, item_index) as usize;
                cms[(to_index, other_index, draw_index)] += 1;
                if from_label_option.is_some() {
                    cms[(from_index, other_index, draw_index)] -= 1;
                }
            }
            self.compute_loss(&state, &cms)
        }
    }

    #[allow(unused_variables)]
    fn decision_callback(
        &mut self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) {
    }
}

// Expectation of the Binder loss

pub struct BinderCMLossComputer {}

impl BinderCMLossComputer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn n_squared(x: CountType) -> f64 {
        let x = x as f64;
        x * x
    }
}

impl CMLossComputer for BinderCMLossComputer {
    fn compute_loss(&self, state: &WorkingClustering, cms: &Array3<CountType>) -> f64 {
        let mut sum: f64 = state
            .occupied_clusters()
            .iter()
            .map(|i| BinderCMLossComputer::n_squared(state.size_of(*i)))
            .sum();
        let n_draws = cms.len_of(Axis(2));
        sum *= n_draws as f64;
        for draw_index in 0..n_draws {
            for other_index in 0..cms.len_of(Axis(1)) {
                let n = cms[(0, other_index, draw_index)];
                if n > 0 {
                    sum += BinderCMLossComputer::n_squared(cms[(0, other_index, draw_index)]);
                    for main_label in state.occupied_clusters().iter() {
                        sum -= 2.0
                            * BinderCMLossComputer::n_squared(
                                cms[(*main_label as usize + 1, other_index, draw_index)],
                            );
                    }
                }
            }
        }
        sum / (n_draws as f64 * BinderCMLossComputer::n_squared(state.n_items()))
    }

    fn change_in_loss(
        &self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) -> f64 {
        let offset = if from_label_option.is_some() && to_label == from_label_option.unwrap() {
            1
        } else {
            0
        };
        let n_draws = cms.len_of(Axis(2));
        let mut sum = (n_draws as f64) * ((state.size_of(to_label) - offset) as f64) / 2.0;
        let to_index = to_label as usize + 1;
        for draw_index in 0..n_draws {
            let other_index = draws.label(draw_index, item_index) as usize;
            sum -= (cms[(to_index, other_index, draw_index)] - offset) as f64;
        }
        sum
    }
}

// Expectation of one minus the adjusted Rand index loss

pub struct OMARICMLossComputer {
    n: CountType,
    sum2: f64,
    sums: Array2<f64>,
}

impl OMARICMLossComputer {
    pub fn new(n_draws: usize) -> Self {
        Self {
            n: 0,
            sum2: 0.0,
            sums: Array2::<f64>::zeros((n_draws, 2)),
        }
    }

    pub fn n_choose_2_times_2(x: CountType) -> f64 {
        let x = x as f64;
        x * (x - 1.0)
    }
}

impl CMLossComputer for OMARICMLossComputer {
    fn initialize(&mut self, state: &WorkingClustering, cms: &Array3<CountType>) {
        self.n = state.n_items();
        self.sum2 = state
            .occupied_clusters()
            .iter()
            .map(|i| OMARICMLossComputer::n_choose_2_times_2(state.size_of(*i)))
            .sum();
        let n_draws = cms.len_of(Axis(2));
        for draw_index in 0..n_draws {
            for other_index in 0..cms.len_of(Axis(1)) {
                let n = cms[(0, other_index, draw_index)];
                if n > 0 {
                    self.sums[(draw_index, 0)] += OMARICMLossComputer::n_choose_2_times_2(n);
                    for main_label in state.occupied_clusters().iter() {
                        self.sums[(draw_index, 1)] += OMARICMLossComputer::n_choose_2_times_2(
                            cms[(*main_label as usize + 1, other_index, draw_index)],
                        );
                    }
                }
            }
        }
    }

    fn compute_loss(&self, _state: &WorkingClustering, _cms: &Array3<CountType>) -> f64 {
        let mut sum = 0.0;
        let sum2 = self.sum2;
        let sum2_div_denom = sum2 / OMARICMLossComputer::n_choose_2_times_2(self.n);
        let n_draws = self.sums.len_of(Axis(0));
        for draw_index in 0..n_draws {
            let sum1 = self.sums[(draw_index, 0)];
            let offset = sum1 * sum2_div_denom;
            let denominator = 0.5 * (sum1 + sum2) - offset;
            if denominator > 0.0 {
                let numerator = self.sums[(draw_index, 1)] - offset;
                sum += numerator / denominator;
            }
        }
        1.0 - sum / (n_draws as f64)
    }

    fn change_in_loss(
        &self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) -> f64 {
        let offset = if from_label_option.is_some() && to_label == from_label_option.unwrap() {
            1
        } else {
            0
        };
        let mut sum2 = if offset == 0 {
            self.sum2 + 2.0 * state.size_of(to_label) as f64
        } else {
            self.sum2
        };
        let to_index = to_label as usize + 1;
        let mut n = self.n;
        let from_index = if from_label_option.is_some() {
            if offset == 0 {
                sum2 -= 2.0 * (state.size_of(from_label_option.unwrap()) - 1) as f64;
            }
            from_label_option.unwrap() as usize + 1
        } else {
            n += 1;
            0
        };
        let sum2_div_denom = sum2 / OMARICMLossComputer::n_choose_2_times_2(n);
        let mut sum = 0.0;
        let n_draws = self.sums.len_of(Axis(0));
        for draw_index in 0..n_draws {
            let other_index = draws.label(draw_index, item_index) as usize;
            let sum1 = self.sums[(draw_index, 0)]
                + if from_label_option.is_none() {
                    2.0 * n as f64
                } else {
                    0.0
                };
            let chance_adjustment = sum1 * sum2_div_denom;
            let denominator = 0.5 * (sum1 + sum2) - chance_adjustment;
            if denominator > 0.0 {
                let mut unadjusted_numerator = self.sums[(draw_index, 1)];
                if offset == 0 {
                    unadjusted_numerator += 2.0 * cms[(to_index, other_index, draw_index)] as f64;
                    if from_label_option.is_some() {
                        unadjusted_numerator -=
                            2.0 * (cms[(from_index, other_index, draw_index)] - 1) as f64;
                    }
                }
                let numerator = unadjusted_numerator - chance_adjustment;
                sum += numerator / denominator;
            }
        }
        1.0 - sum / (n_draws as f64)
    }

    fn decision_callback(
        &mut self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) {
        self.sum2 += 2.0 * state.size_of(to_label) as f64;
        let to_index = to_label as usize + 1;
        let from_index = if from_label_option.is_some() {
            self.sum2 -= 2.0 * (state.size_of(from_label_option.unwrap()) - 1) as f64;
            from_label_option.unwrap() as usize + 1
        } else {
            self.n += 1;
            0
        };
        let n_draws = cms.len_of(Axis(2));
        for draw_index in 0..n_draws {
            let other_index = draws.label(draw_index, item_index) as usize;
            let n = cms[(0, other_index, draw_index)];
            if n > 0 {
                if from_label_option.is_some() {
                    self.sums[(draw_index, 1)] -=
                        2.0 * (cms[(from_index, other_index, draw_index)] - 1) as f64;
                } else {
                    self.sums[(draw_index, 0)] += 2.0 * n as f64;
                }
                self.sums[(draw_index, 1)] += 2.0 * cms[(to_index, other_index, draw_index)] as f64;
            }
        }
    }
}

// Expectation of variation of information loss

pub struct VICMLossComputer<'a> {
    cache: &'a Log2Cache,
}

impl<'a> VICMLossComputer<'a> {
    pub fn new(cache: &'a Log2Cache) -> Self {
        Self { cache }
    }
}

impl<'a> CMLossComputer for VICMLossComputer<'a> {
    fn compute_loss(&self, state: &WorkingClustering, cms: &Array3<CountType>) -> f64 {
        let sum2: f64 = state
            .occupied_clusters()
            .iter()
            .map(|i| self.cache.nlog2n(state.size_of(*i)))
            .sum();
        let n_draws = cms.len_of(Axis(2));
        let mut sum = 0.0;
        for draw_index in 0..n_draws {
            let mut vi = 0.0;
            for other_index in 0..cms.len_of(Axis(1)) {
                let n = cms[(0, other_index, draw_index)];
                if n > 0 {
                    vi += self.cache.nlog2n(cms[(0, other_index, draw_index)]);
                    for main_label in state.occupied_clusters().iter() {
                        vi -= 2.0
                            * self
                                .cache
                                .nlog2n(cms[(*main_label as usize + 1, other_index, draw_index)]);
                    }
                }
            }
            sum += (vi + sum2) / (state.n_items() as f64);
        }
        sum / (n_draws as f64)
    }

    fn change_in_loss(
        &self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) -> f64 {
        let offset = if from_label_option.is_some() && to_label == from_label_option.unwrap() {
            1
        } else {
            0
        };
        let n_draws = cms.len_of(Axis(2));
        let mut sum = (n_draws as f64)
            * self
                .cache
                .nlog2n_difference(state.size_of(to_label) - offset)
            / 2.0;
        let to_index = to_label as usize + 1;
        for draw_index in 0..n_draws {
            let other_index = draws.label(draw_index, item_index) as usize;
            let n12 = cms[(to_index, other_index, draw_index)] - offset;
            sum -= self.cache.nlog2n_difference(n12);
        }
        sum
    }
}

// Common

fn find_label_of_minimum<I: Iterator<Item = (LabelType, f64)>>(pairs: I) -> LabelType {
    let mut s0 = f64::INFINITY;
    let mut l0 = 0;
    for pair in pairs {
        if pair.1 < s0 {
            s0 = pair.1;
            l0 = pair.0;
        }
    }
    l0
}

fn allocation_scan<T: CMLossComputer>(
    sweetening_scan: bool,
    singletons_initialization: bool,
    state: &mut WorkingClustering,
    cms: &mut Array3<CountType>,
    permutation: &Vec<usize>,
    loss_computer: &mut T,
    draws: &Clusterings,
) -> bool {
    let mut state_changed = false;
    for item_index in permutation {
        let item_index = *item_index;
        let label_of_empty_cluster = state.label_of_empty_cluster();
        let from_label_option = match sweetening_scan {
            true => Some(state.get(item_index)),
            false => None,
        };
        let to_label =
            if !sweetening_scan && singletons_initialization && label_of_empty_cluster.is_some() {
                label_of_empty_cluster.unwrap()
            } else {
                let iter = state
                    .occupied_clusters()
                    .iter()
                    .chain(label_of_empty_cluster.iter())
                    .map(|to_label| {
                        (
                            *to_label,
                            loss_computer.change_in_loss(
                                item_index,
                                *to_label,
                                from_label_option,
                                &state,
                                &cms,
                                &draws,
                            ),
                        )
                    });
                find_label_of_minimum(iter)
            };
        if !sweetening_scan || to_label != from_label_option.unwrap() {
            loss_computer.decision_callback(
                item_index,
                to_label,
                from_label_option,
                &state,
                &cms,
                &draws,
            );
            if sweetening_scan {
                state.reassign(item_index, to_label);
            } else {
                state.assign(item_index, to_label);
            }
            let from_index = if sweetening_scan {
                from_label_option.unwrap() as usize + 1
            } else {
                0
            };
            let to_index = to_label as usize + 1;
            for draw_index in 0..draws.n_clusterings() {
                let other_index = draws.label(draw_index, item_index) as usize;
                if sweetening_scan {
                    cms[(from_index, other_index, draw_index)] -= 1;
                } else {
                    cms[(0, other_index, draw_index)] += 1;
                }
                cms[(to_index, other_index, draw_index)] += 1;
            }
            state_changed = true;
        }
    }
    state_changed
}

fn constrained_allocation_scan<T: CMLossComputer>(
    state: &mut WorkingClustering,
    cms: &mut Array3<CountType>,
    label1: LabelType,
    label2: LabelType,
    permutation: &Vec<usize>,
    loss_computer: &mut T,
    draws: &Clusterings,
) -> bool {
    let mut first_is_empty = true;
    let mut second_is_empty = true;
    for item_index in permutation {
        let item_index = *item_index;
        let to_label = {
            let possibilities = [label1, label2];
            let iter = possibilities.iter().map(|to_label| {
                (
                    *to_label,
                    loss_computer.change_in_loss(item_index, *to_label, None, &state, &cms, &draws),
                )
            });
            find_label_of_minimum(iter)
        };
        if to_label == label1 {
            first_is_empty = false
        } else {
            second_is_empty = false
        }
        loss_computer.decision_callback(item_index, to_label, None, &state, &cms, &draws);
        state.assign(item_index, to_label);
        let to_index = to_label as usize + 1;
        for draw_index in 0..draws.n_clusterings() {
            let other_index = draws.label(draw_index, item_index) as usize;
            cms[(0, other_index, draw_index)] += 1;
            cms[(to_index, other_index, draw_index)] += 1;
        }
    }
    !first_is_empty && !second_is_empty
}

pub fn minimize_once_by_salso_v2<'a, T: CMLossComputer, U: Rng>(
    loss_computer_factory: Box<dyn Fn() -> T + 'a>,
    draws: &Clusterings,
    p: SALSOParameters,
    stop_time: &SystemTime,
    rng: &mut U,
) -> SALSOResults {
    let n_items = draws.n_items();
    let max_size = match p.max_size {
        0 => draws.max_clusters(),
        _ => p.max_size,
    };
    let mut permutation: Vec<usize> = (0..p.n_items).collect();
    let mut best = SALSOResults::dummy(max_size);
    for run_counter in 1..=p.n_runs {
        let mut loss_computer = loss_computer_factory();
        let (mut state, mut cms, initialization_method) =
            if rng.gen_range(0.0, 1.0) < p.prob_sequential_allocation {
                let mut state = WorkingClustering::empty(n_items, max_size);
                let mut cms = Array3::<CountType>::zeros((
                    max_size as usize + 1,
                    draws.max_clusters() as usize,
                    draws.n_clusterings(),
                ));
                // Sequential allocation
                permutation.shuffle(rng);
                let singletons_initialization =
                    rng.gen_range(0.0, 1.0) < p.prob_singletons_initialization;
                allocation_scan(
                    false,
                    singletons_initialization,
                    &mut state,
                    &mut cms,
                    &permutation,
                    &mut loss_computer,
                    draws,
                );
                let initialization_method = if singletons_initialization {
                    InitializationMethod::SequentialFromSingletons
                } else {
                    InitializationMethod::SequentialFromEmpty
                };
                (state, cms, initialization_method)
            } else {
                let state = WorkingClustering::random(n_items, max_size, rng);
                let cms = draws.make_confusion_matrices(&state);
                loss_computer.initialize(&state, &cms);
                (
                    state,
                    cms,
                    InitializationMethod::SampleOne2MaxWithReplacement,
                )
            };
        let (mut scan_counter, mut merge_counter, mut split_counter) = (0, 0, 0);
        let mut expected_loss_option = None;
        'bigloop: loop {
            // Sweetening scans
            let mut state_changed = true;
            while state_changed && scan_counter < p.max_scans {
                scan_counter += 1;
                permutation.shuffle(rng);
                state_changed = allocation_scan(
                    true,
                    false,
                    &mut state,
                    &mut cms,
                    &permutation,
                    &mut loss_computer,
                    draws,
                );
                if state_changed {
                    expected_loss_option = None;
                }
            }
            if expected_loss_option.is_none() {
                expected_loss_option = Some(loss_computer.compute_loss(&state, &cms));
            }
            if p.merge_split_strategy == 1
                || state.occupied_clusters().len() <= p.merge_split_strategy as usize
            {
                // Try merging
                for label1 in state.occupied_clusters() {
                    let s1 = state.size_of(*label1);
                    if s1 < 2 {
                        continue;
                    }
                    for label2 in state.occupied_clusters() {
                        if *label1 == *label2 {
                            continue;
                        }
                        let s2 = state.size_of(*label2);
                        if s2 < 2 {
                            continue;
                        }
                        // Copying is cheaper than changing (and undoing) the original state.
                        let mut state2 = state.clone();
                        let mut cms2 = cms.clone();
                        // Drain the cluster with the fewer items.
                        let (label1, label2, s2) = if s1 < s2 {
                            (label2, label1, s1)
                        } else {
                            (label1, label2, s2)
                        };
                        let to_index = *label1 as usize + 1;
                        let from_index = *label2 as usize + 1;
                        let mut hit_counter = 0;
                        for item_index in 0..(state2.n_items() as usize) {
                            if state2.get(item_index) == *label2 {
                                state2.reassign(item_index, *label1);
                                for draw_index in 0..draws.n_clusterings() {
                                    let other_index = draws.label(draw_index, item_index) as usize;
                                    cms2[(from_index, other_index, draw_index)] -= 1;
                                    cms2[(to_index, other_index, draw_index)] += 1;
                                }
                                hit_counter += 1;
                                if hit_counter == s2 {
                                    break;
                                }
                            }
                        }
                        let expected_loss2 = loss_computer.compute_loss(&state2, &cms2);
                        if expected_loss2 < expected_loss_option.unwrap() {
                            merge_counter += 1;
                            state = state2;
                            cms = cms2;
                            expected_loss_option = Some(expected_loss2);
                            continue 'bigloop;
                        }
                    }
                }
                // Try splitting
                let label_of_empty_cluster = state.label_of_empty_cluster();
                if label_of_empty_cluster.is_some() {
                    for label in state.occupied_clusters() {
                        let s2 = state.size_of(*label) as usize;
                        if s2 < 4 {
                            continue;
                        }
                        let mut state2 = state.clone();
                        let mut cms2 = cms.clone();
                        let from_index = *label as usize + 1;
                        let mut active_items = Vec::with_capacity(s2);
                        for item_index in 0..(state2.n_items() as usize) {
                            if state2.get(item_index) == *label {
                                state2.remove(item_index);
                                for draw_index in 0..draws.n_clusterings() {
                                    let other_index = draws.label(draw_index, item_index) as usize;
                                    cms2[(0, other_index, draw_index)] -= 1;
                                    cms2[(from_index, other_index, draw_index)] -= 1;
                                }
                                active_items.push(item_index);
                                if active_items.len() == s2 {
                                    break;
                                }
                            }
                        }
                        active_items.shuffle(rng);
                        let both_nonempty = constrained_allocation_scan(
                            &mut state2,
                            &mut cms2,
                            *label,
                            label_of_empty_cluster.unwrap(),
                            &active_items,
                            &mut loss_computer,
                            draws,
                        );
                        if both_nonempty {
                            let expected_loss2 = loss_computer.compute_loss(&state2, &cms2);
                            if expected_loss2 < expected_loss_option.unwrap() {
                                split_counter += 1;
                                state = state2;
                                cms = cms2;
                                expected_loss_option = Some(expected_loss2);
                                continue 'bigloop;
                            }
                        }
                    }
                }
            }
            break;
        }
        // Tidy up
        if expected_loss_option.unwrap() < best.expected_loss {
            let clustering = state.standardize().iter().map(|x| *x as usize).collect();
            best = SALSOResults {
                clustering,
                expected_loss: expected_loss_option.unwrap(),
                n_scans: scan_counter,
                n_merges: merge_counter,
                n_splits: split_counter,
                initialization_method,
                ..best
            }
        }
        if SystemTime::now() > *stop_time {
            return SALSOResults {
                n_runs: run_counter,
                ..best
            };
        }
    }
    SALSOResults {
        n_runs: p.n_runs,
        ..best
    }
}

// ******************************************
// Implementation of SALSO for general losses
// ******************************************

pub trait GeneralLossComputer {
    fn expected_loss_kernel(&self) -> f64;
    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: LabelType) -> f64;
    fn new_subset(&mut self, partition: &mut Partition);
    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType);
    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType;
}

// Expectation of the Binder loss

#[derive(Debug)]
struct BinderSubsetCalculations {
    committed_loss: f64,
    speculative_loss: f64,
}

pub struct BinderGLossComputer<'a> {
    subsets: Vec<BinderSubsetCalculations>,
    psm: &'a SquareMatrixBorrower<'a>,
}

impl<'a> BinderGLossComputer<'a> {
    pub fn new(psm: &'a SquareMatrixBorrower<'a>) -> Self {
        Self {
            subsets: Vec::new(),
            psm,
        }
    }
    fn expected_loss_from_kernel(psm: &'a SquareMatrixBorrower<'a>, kernel: f64) -> f64 {
        let nif = psm.n_items() as f64;
        (2.0 * kernel + psm.sum_of_triangle()) * 2.0 / (nif * nif)
    }
}

impl<'a> GeneralLossComputer for BinderGLossComputer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        self.subsets
            .iter()
            .fold(0.0, |s, subset| s + subset.committed_loss)
    }

    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: LabelType) -> f64 {
        self.subsets[subset_index as usize].speculative_loss = partition.subsets()
            [subset_index as usize]
            .items()
            .iter()
            .fold(0.0, |s, j| {
                s + 0.5 - unsafe { *self.psm.get_unchecked((i, *j)) }
            });
        self.subsets[subset_index as usize].speculative_loss
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(BinderSubsetCalculations {
            committed_loss: 0.0,
            speculative_loss: 0.0,
        });
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        self.subsets[subset_index as usize].committed_loss +=
            self.subsets[subset_index as usize].speculative_loss;
        partition.add_with_index(i, subset_index as usize);
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        let subset_index = partition.label_of(i).unwrap();
        self.subsets[subset_index].committed_loss -= partition.subsets()[subset_index]
            .items()
            .iter()
            .fold(0.0, |s, j| {
                let jj = *j;
                s + if jj != i {
                    0.5 - unsafe { *self.psm.get_unchecked((i, jj)) }
                } else {
                    0.0
                }
            });
        partition.remove_clean_and_relabel(i, |killed_subset_index, moved_subset_index| {
            self.subsets.swap_remove(killed_subset_index);
            assert_eq!(moved_subset_index, self.subsets.len());
        });
        subset_index as LabelType
    }
}

// First-order approximation of expectation of one minus the adjusted Rand index loss

#[derive(Debug)]
struct OMARIApproxSubsetCalculations {
    committed_ip: f64,
    committed_i: f64,
    speculative_ip: f64,
    speculative_i: f64,
}

pub struct OMARIApproxGLossComputer<'a> {
    committed_n_items: usize,
    committed_sum_psm: f64,
    speculative_sum_psm: f64,
    subsets: Vec<OMARIApproxSubsetCalculations>,
    psm: &'a SquareMatrixBorrower<'a>,
}

impl<'a> OMARIApproxGLossComputer<'a> {
    pub fn new(psm: &'a SquareMatrixBorrower<'a>) -> Self {
        Self {
            committed_n_items: 0,
            committed_sum_psm: 0.0,
            speculative_sum_psm: f64::NEG_INFINITY,
            subsets: Vec::new(),
            psm,
        }
    }
}

impl<'a> OMARIApproxGLossComputer<'a> {
    fn engine(
        &self,
        speculative_n_items: usize,
        speculative_ip: f64,
        speculative_i: f64,
        speculative_sum_psm: f64,
    ) -> f64 {
        let n_items = speculative_n_items + self.committed_n_items;
        if n_items <= 1 {
            return f64::INFINITY;
        }
        let n_choose_2 = (n_items * (n_items - 1) / 2) as f64;
        let all_ip = speculative_ip + self.subsets.iter().fold(0.0, |s, c| s + c.committed_ip);
        let all_i = speculative_i + self.subsets.iter().fold(0.0, |s, c| s + c.committed_i);
        let all_p = speculative_sum_psm + self.committed_sum_psm;
        let subtractor = all_i * all_p / n_choose_2;
        let numerator = all_ip - subtractor;
        let denominator = 0.5 * (all_i + all_p) - subtractor;
        1.0 - numerator / denominator
    }
}

impl<'a> GeneralLossComputer for OMARIApproxGLossComputer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        self.engine(0, 0.0, 0.0, 0.0)
    }

    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: LabelType) -> f64 {
        let s = &partition.subsets()[subset_index as usize];
        self.subsets[subset_index as usize].speculative_ip = s
            .items()
            .iter()
            .fold(0.0, |s, j| s + unsafe { *self.psm.get_unchecked((i, *j)) });
        self.subsets[subset_index as usize].speculative_i = s.n_items() as f64;
        if self.speculative_sum_psm == f64::NEG_INFINITY {
            self.speculative_sum_psm = partition.subsets().iter().fold(0.0, |s, subset| {
                // We use the NEG_INFINITY flag to see if we need to do the computation.
                s + subset.items().iter().fold(0.0, |ss, j| {
                    ss + unsafe { *self.psm.get_unchecked((i, *j)) }
                })
            })
        }
        self.engine(
            1,
            self.subsets[subset_index as usize].speculative_ip,
            self.subsets[subset_index as usize].speculative_i,
            self.speculative_sum_psm,
        )
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(OMARIApproxSubsetCalculations {
            committed_ip: 0.0,
            committed_i: 0.0,
            speculative_ip: 0.0,
            speculative_i: 0.0,
        });
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        let mut sc = &mut self.subsets[subset_index as usize];
        sc.committed_ip += sc.speculative_ip;
        sc.committed_i += sc.speculative_i;
        self.committed_n_items += 1;
        self.committed_sum_psm += self.speculative_sum_psm;
        self.speculative_sum_psm = f64::NEG_INFINITY;
        partition.add_with_index(i, subset_index as usize);
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        let subset_index = partition.label_of(i).unwrap();
        self.subsets[subset_index].committed_ip -= partition.subsets()[subset_index]
            .items()
            .iter()
            .fold(0.0, |s, j| {
                let jj = *j;
                s + if jj != i {
                    unsafe { *self.psm.get_unchecked((i, jj)) }
                } else {
                    0.0
                }
            });
        self.subsets[subset_index].committed_i -=
            (partition.subsets()[subset_index].n_items() - 1) as f64;
        self.committed_n_items -= 1;
        self.committed_sum_psm -= partition.subsets().iter().fold(0.0, |s, subset| {
            // We use the NEG_INFINITY flag to see if we need to do the computation.
            s + subset.items().iter().fold(0.0, |ss, j| {
                let jj = *j;
                ss + if jj != i {
                    unsafe { *self.psm.get_unchecked((i, jj)) }
                } else {
                    0.0
                }
            })
        });
        partition.remove_clean_and_relabel(i, |killed_subset_index, moved_subset_index| {
            self.subsets.swap_remove(killed_subset_index);
            assert_eq!(moved_subset_index, self.subsets.len());
        });
        subset_index as LabelType
    }
}

// Lower bound of the expectation of variation of information loss

#[derive(Debug)]
struct VILBCacheUnit {
    item: usize,
    committed_sum: f64,
    committed_contribution: f64,
    speculative_sum: f64,
    speculative_contribution: f64,
}

#[derive(Debug)]
struct VILBSubsetCalculations {
    cached_units: Vec<VILBCacheUnit>,
    committed_loss: f64,
    speculative_loss: f64,
}

pub struct VILBGLossComputer<'a> {
    subsets: Vec<VILBSubsetCalculations>,
    psm: &'a SquareMatrixBorrower<'a>,
}

impl<'a> VILBGLossComputer<'a> {
    pub fn new(psm: &'a SquareMatrixBorrower<'a>) -> Self {
        Self {
            subsets: Vec::new(),
            psm,
        }
    }
    pub fn expected_loss_from_kernel(psm: &'a SquareMatrixBorrower<'a>, kernel: f64) -> f64 {
        let nif = psm.n_items() as f64;
        (kernel + vilb_expected_loss_constant(psm)) / nif
    }
}

impl<'a> GeneralLossComputer for VILBGLossComputer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        self.subsets
            .iter()
            .fold(0.0, |s, subset| s + subset.committed_loss)
    }

    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: LabelType) -> f64 {
        let subset_of_partition = &partition.subsets()[subset_index as usize];
        if subset_of_partition.n_items() == 0 {
            self.subsets[subset_index as usize]
                .cached_units
                .push(VILBCacheUnit {
                    item: i,
                    committed_sum: 0.0,
                    committed_contribution: 0.0,
                    speculative_sum: 1.0,
                    speculative_contribution: 0.0,
                });
            return 0.0;
        }
        for cu in self.subsets[subset_index as usize].cached_units.iter_mut() {
            cu.speculative_sum =
                cu.committed_sum + unsafe { *self.psm.get_unchecked((cu.item, i)) };
            cu.speculative_contribution = cu.speculative_sum.log2();
        }
        let sum = subset_of_partition
            .items()
            .iter()
            .fold(0.0, |s, j| s + unsafe { *self.psm.get_unchecked((i, *j)) })
            + 1.0; // Because self.psm[(i, i)] == 1;
        self.subsets[subset_index as usize]
            .cached_units
            .push(VILBCacheUnit {
                item: i,
                committed_sum: 0.0,
                committed_contribution: 0.0,
                speculative_sum: sum,
                speculative_contribution: sum.log2(),
            });
        let nif = subset_of_partition.n_items() as f64;
        let s1 = (nif + 1.0) * (nif + 1.0).log2();
        let s2 = self.subsets[subset_index as usize]
            .cached_units
            .iter()
            .fold(0.0, |s, cu| s + cu.speculative_contribution);
        self.subsets[subset_index as usize].speculative_loss = s1 - 2.0 * s2;
        self.subsets[subset_index as usize].speculative_loss
            - self.subsets[subset_index as usize].committed_loss
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(VILBSubsetCalculations {
            cached_units: Vec::new(),
            committed_loss: 0.0,
            speculative_loss: 0.0,
        })
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        for (index, subset) in self.subsets.iter_mut().enumerate() {
            if index == (subset_index as usize) {
                for cu in subset.cached_units.iter_mut() {
                    cu.committed_sum = cu.speculative_sum;
                    cu.committed_contribution = cu.speculative_contribution;
                }
            } else {
                subset.cached_units.pop();
            }
        }
        self.subsets[subset_index as usize].committed_loss =
            self.subsets[subset_index as usize].speculative_loss;
        partition.add_with_index(i, subset_index as usize);
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        let subset_index = partition.label_of(i).unwrap();
        for cu in self.subsets[subset_index].cached_units.iter_mut() {
            cu.committed_sum -= unsafe { *self.psm.get_unchecked((cu.item, i)) };
            cu.committed_contribution = cu.committed_sum.log2();
        }
        let pos = self.subsets[subset_index]
            .cached_units
            .iter()
            .enumerate()
            .find(|cu| cu.1.item == i)
            .unwrap()
            .0;
        self.subsets[subset_index].cached_units.swap_remove(pos);
        self.subsets[subset_index].committed_loss =
            match partition.subsets()[subset_index].n_items() {
                0 => 0.0,
                ni => {
                    let nif = ni as f64;
                    nif * nif.log2()
                        - 2.0
                            * self.subsets[subset_index]
                                .cached_units
                                .iter()
                                .fold(0.0, |s, cu| s + cu.committed_contribution)
                }
            };
        partition.remove_clean_and_relabel(i, |killed_subset_index, moved_subset_index| {
            self.subsets.swap_remove(killed_subset_index);
            assert_eq!(moved_subset_index, self.subsets.len());
        });
        subset_index as LabelType
    }
}

// Common

fn label_of_empty_cluster<U: GeneralLossComputer>(
    partition: &mut Partition,
    computer: &mut U,
    max_label: LabelType,
) -> Option<LabelType> {
    match partition.subsets().last() {
        None => {
            computer.new_subset(partition);
            Some(0)
        }
        Some(last) => {
            if last.is_empty() {
                Some(partition.n_subsets() as LabelType - 1)
            } else {
                if partition.n_subsets() <= max_label as usize {
                    computer.new_subset(partition);
                    Some(partition.n_subsets() as LabelType - 1)
                } else {
                    None
                }
            }
        }
    }
}

fn micro_optimized_allocation<U: GeneralLossComputer>(
    partition: &mut Partition,
    computer: &mut U,
    i: usize,
    label_to_take: Option<LabelType>,
) -> LabelType {
    let max_label = partition.n_subsets() as LabelType - 1;
    let iter = (0..=max_label).map(|subset_index| {
        let value = computer.speculative_add(partition, i, subset_index as LabelType);
        if label_to_take.is_some() {
            if subset_index == label_to_take.unwrap() {
                (subset_index, f64::NEG_INFINITY)
            } else {
                (subset_index, f64::INFINITY)
            }
        } else {
            (subset_index, value)
        }
    });
    let subset_index = find_label_of_minimum(iter);
    computer.add_with_index(partition, i, subset_index);
    subset_index as LabelType
}

pub fn minimize_once_by_salso<'a, T: Rng, U: GeneralLossComputer>(
    computer_factory: Box<dyn Fn() -> U + 'a>,
    p: SALSOParameters,
    stop_time: &SystemTime,
    rng: &mut T,
) -> SALSOResults {
    let max_label = p.max_size.max(1) - 1;
    let mut best = SALSOResults::dummy(p.max_size);
    let mut permutation: Vec<usize> = (0..p.n_items).collect();
    let mut run_counter = 0;
    while run_counter < p.n_runs {
        let mut computer = computer_factory();
        let (mut partition, initialization_method) = if rng.gen_range(0.0, 1.0)
            < p.prob_sequential_allocation
        {
            let mut partition = Partition::new(p.n_items);
            permutation.shuffle(rng);
            let singletons_initialization =
                rng.gen_range(0.0, 1.0) < p.prob_singletons_initialization;
            // Sequential allocation
            for i in 0..p.n_items {
                let ii = unsafe { *permutation.get_unchecked(i) };
                let empty_label = label_of_empty_cluster(&mut partition, &mut computer, max_label);
                if singletons_initialization {
                    micro_optimized_allocation(&mut partition, &mut computer, ii, empty_label);
                } else {
                    micro_optimized_allocation(&mut partition, &mut computer, ii, None);
                }
            }
            let initialization_method = if singletons_initialization {
                InitializationMethod::SequentialFromSingletons
            } else {
                InitializationMethod::SequentialFromEmpty
            };
            (partition, initialization_method)
        } else {
            let mut partition = Partition::new(p.n_items);
            let destiny = {
                let mut v = Vec::with_capacity(p.n_items);
                v.resize_with(p.n_items, || rng.gen_range(0, p.max_size));
                Partition::from(&v) // Already canonicalized
            };
            for i in 0..p.n_items {
                let label_to_take = Some(destiny.label_of(i).unwrap() as LabelType);
                label_of_empty_cluster(&mut partition, &mut computer, max_label);
                micro_optimized_allocation(&mut partition, &mut computer, i, label_to_take);
            }
            (
                partition,
                InitializationMethod::SampleOne2MaxWithReplacement,
            )
        };
        // Sweetening scans
        let mut n_scans = 0;
        let mut change = true;
        while change && n_scans < p.max_scans {
            change = false;
            n_scans += 1;
            permutation.shuffle(rng);
            for i in 0..p.n_items {
                label_of_empty_cluster(&mut partition, &mut computer, max_label);
                let ii = unsafe { *permutation.get_unchecked(i) };
                let previous_subset_index = computer.remove(&mut partition, ii);
                let subset_index =
                    micro_optimized_allocation(&mut partition, &mut computer, ii, None);
                if subset_index != previous_subset_index {
                    change = true;
                };
            }
        }
        let expected_loss = computer.expected_loss_kernel();
        if expected_loss < best.expected_loss {
            partition.canonicalize();
            best = SALSOResults {
                clustering: partition.labels_via_copying(),
                expected_loss,
                n_scans,
                initialization_method,
                ..best
            };
        }
        run_counter += 1;
        if SystemTime::now() > *stop_time {
            return SALSOResults {
                n_runs: run_counter,
                ..best
            };
        }
    }
    SALSOResults {
        n_runs: p.n_runs,
        ..best
    }
}

// Common implementation for SALSO.

#[derive(Debug, Copy, Clone)]
pub struct SALSOParameters {
    n_items: usize,
    max_size: LabelType,
    max_scans: u32,
    n_runs: u32,
    prob_sequential_allocation: f64,
    prob_singletons_initialization: f64,
    merge_split_strategy: LabelType,
}

pub struct SALSOResults {
    clustering: Vec<usize>,
    expected_loss: f64,
    n_scans: u32,
    n_merges: u32,
    n_splits: u32,
    initialization_method: InitializationMethod,
    n_runs: u32,
    max_size: LabelType,
}

impl SALSOResults {
    pub fn new(
        clustering: Vec<usize>,
        expected_loss: f64,
        n_scans: u32,
        n_merges: u32,
        n_splits: u32,
        initialization_method: InitializationMethod,
        n_runs: u32,
        max_size: LabelType,
    ) -> Self {
        Self {
            clustering,
            expected_loss,
            n_scans,
            n_merges,
            n_splits,
            initialization_method,
            n_runs,
            max_size,
        }
    }

    pub fn dummy(max_size: LabelType) -> Self {
        SALSOResults::new(
            vec![0usize; 0],
            std::f64::INFINITY,
            0,
            0,
            0,
            InitializationMethod::SequentialFromEmpty,
            0,
            max_size,
        )
    }
}

pub fn minimize_by_salso<T: Rng>(
    pdi: PartitionDistributionInformation,
    loss_function: LossFunction,
    p: SALSOParameters,
    seconds: u64,
    nanoseconds: u32,
    parallel: bool,
    mut rng: &mut T,
) -> SALSOResults {
    let cache = Log2Cache::new(if let LossFunction::VI = loss_function {
        p.n_items
    } else {
        0
    });
    let stop_time = SystemTime::now() + Duration::new(seconds, nanoseconds);
    let result = if !parallel {
        match loss_function {
            LossFunction::BinderDraws => minimize_once_by_salso_v2(
                Box::new(|| BinderCMLossComputer::new()),
                pdi.draws(),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::BinderPSM => minimize_once_by_salso(
                Box::new(|| BinderGLossComputer::new(pdi.psm())),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::OneMinusARI => minimize_once_by_salso_v2(
                Box::new(|| OMARICMLossComputer::new(pdi.draws().n_clusterings())),
                pdi.draws(),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::OneMinusARIapprox => minimize_once_by_salso(
                Box::new(|| OMARIApproxGLossComputer::new(pdi.psm())),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::VI => minimize_once_by_salso_v2(
                Box::new(|| VICMLossComputer::new(&cache)),
                pdi.draws(),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::VIlb => minimize_once_by_salso(
                Box::new(|| VILBGLossComputer::new(pdi.psm())),
                p,
                &stop_time,
                rng,
            ),
        }
    } else {
        let (tx, rx) = mpsc::channel();
        let n_cores = num_cpus::get() as u32;
        let p = SALSOParameters {
            n_runs: (p.n_runs + n_cores - 1) / n_cores,
            ..p
        };
        let cache_ref = &cache;
        crossbeam::scope(|s| {
            for _ in 0..n_cores {
                let tx = mpsc::Sender::clone(&tx);
                let mut child_rng = IsaacRng::from_rng(&mut rng).unwrap();
                s.spawn(move |_| {
                    let result = match loss_function {
                        LossFunction::BinderDraws => minimize_once_by_salso_v2(
                            Box::new(|| BinderCMLossComputer::new()),
                            pdi.draws(),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::BinderPSM => minimize_once_by_salso(
                            Box::new(|| BinderGLossComputer::new(pdi.psm())),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::OneMinusARI => minimize_once_by_salso_v2(
                            Box::new(|| OMARICMLossComputer::new(pdi.draws().n_clusterings())),
                            pdi.draws(),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::OneMinusARIapprox => minimize_once_by_salso(
                            Box::new(|| OMARIApproxGLossComputer::new(pdi.psm())),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::VI => minimize_once_by_salso_v2(
                            Box::new(|| VICMLossComputer::new(cache_ref)),
                            pdi.draws(),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::VIlb => minimize_once_by_salso(
                            Box::new(|| VILBGLossComputer::new(pdi.psm())),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                    };
                    tx.send(result).unwrap();
                });
            }
        })
        .unwrap();
        std::mem::drop(tx); // Because of the cloning in the loop.
        let mut best = SALSOResults::dummy(p.max_size);
        let mut run_counter = 0;
        for candidate in rx {
            run_counter += candidate.n_runs;
            if candidate.expected_loss < best.expected_loss {
                best = candidate;
            }
        }
        best.n_runs = run_counter;
        best
    };
    let result = SALSOResults {
        expected_loss: match loss_function {
            LossFunction::BinderPSM => {
                BinderGLossComputer::expected_loss_from_kernel(pdi.psm(), result.expected_loss)
            }
            LossFunction::VIlb => {
                VILBGLossComputer::expected_loss_from_kernel(pdi.psm(), result.expected_loss)
            }
            _ => result.expected_loss,
        },
        ..result
    };
    result
}

pub fn minimize_by_enumeration(
    f: fn(&[LabelType], &SquareMatrixBorrower) -> f64,
    psm: &SquareMatrixBorrower,
) -> Vec<usize> {
    let (tx, rx) = mpsc::channel();
    crossbeam::scope(|s| {
        for iter in Partition::iter_sharded(num_cpus::get() as u32, psm.n_items()) {
            let tx = mpsc::Sender::clone(&tx);
            s.spawn(move |_| {
                let mut working_minimum = std::f64::INFINITY;
                let mut working_minimizer = vec![0usize; psm.n_items()];
                for partition in iter {
                    let part: Vec<LabelType> = partition.iter().map(|x| *x as LabelType).collect();
                    let value = f(&part[..], psm);
                    if value < working_minimum {
                        working_minimum = value;
                        working_minimizer = partition;
                    }
                }
                tx.send(working_minimizer).unwrap();
            });
        }
    })
    .unwrap();
    std::mem::drop(tx); // Because of the cloning in the loop.
    let mut working_minimum = std::f64::INFINITY;
    let mut working_minimizer = vec![0usize; psm.n_items()];
    for partition in rx {
        let part: Vec<LabelType> = partition.iter().map(|x| *x as LabelType).collect();
        let value = f(&part[..], psm);
        if value < working_minimum {
            working_minimum = value;
            working_minimizer = partition;
        }
    }
    working_minimizer
}

#[cfg(test)]
mod tests_optimize {
    use super::rand::thread_rng;
    use super::*;

    #[test]
    fn test_max_scan() {
        let n_items = 5;
        let mut psm = SquareMatrix::identity(n_items);
        let psm_view = &psm.view();
        let p = SALSOParameters {
            n_items,
            max_size: 2,
            max_scans: 10,
            n_runs: 100,
            prob_sequential_allocation: 0.25,
            prob_singletons_initialization: 0.5,
            merge_split_strategy: 1,
        };
        minimize_by_salso(
            PartitionDistributionInformation::PairwiseSimilarityMatrix(psm_view),
            LossFunction::VIlb,
            p,
            5,
            0,
            false,
            &mut thread_rng(),
        );
    }
}

// API for R

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__minimize_by_salso(
    n_items: i32,
    n_draws: i32,
    draws_ptr: *mut i32,
    psm_ptr: *mut f64,
    loss: i32,
    max_size: i32,
    max_scans: i32,
    n_runs: i32,
    prob_sequential_allocation: f64,
    prob_singletons_initialization: f64,
    merge_split_strategy: i32,
    seconds: f64,
    parallel: i32,
    results_labels_ptr: *mut i32,
    results_expected_loss_ptr: *mut f64,
    results_n_scans_ptr: *mut i32,
    results_n_merges_ptr: *mut i32,
    results_n_splits_ptr: *mut i32,
    results_n_runs_ptr: *mut i32,
    results_max_size_ptr: *mut i32,
    results_initialization_method_ptr: *mut i32,
    seed_ptr: *const i32, // Assumed length is 32
) {
    let n_items = usize::try_from(n_items).unwrap();
    let nd = usize::try_from(n_draws).unwrap();
    let draws = Clusterings::from_i32_column_major_order(
        PartitionsHolderBorrower::from_ptr(draws_ptr, nd, n_items, true).data(),
        n_items,
    );
    let psm = SquareMatrixBorrower::from_ptr(psm_ptr, n_items);
    let max_size = LabelType::try_from(max_size).unwrap();
    let max_scans = u32::try_from(max_scans).unwrap();
    let n_runs = u32::try_from(n_runs).unwrap();
    let merge_split_strategy = LabelType::try_from(merge_split_strategy).unwrap();
    let (secs, nanos) = if seconds.is_infinite() || seconds < 0.0 {
        (1000 * 365 * 24 * 60 * 60, 0) // 1,000 years
    } else {
        (
            seconds.floor() as u64,
            ((seconds - seconds.floor()) * 1_000_000_000.0).floor() as u32,
        )
    };
    let parallel = parallel != 0;
    let mut rng = mk_rng_isaac(seed_ptr);
    let (loss_function, pdi) = match LossFunction::from_code(loss) {
        Some(loss_function) => match loss_function {
            LossFunction::BinderDraws | LossFunction::OneMinusARI | LossFunction::VI => (
                loss_function,
                PartitionDistributionInformation::Draws(&draws),
            ),
            LossFunction::BinderPSM | LossFunction::OneMinusARIapprox | LossFunction::VIlb => (
                loss_function,
                PartitionDistributionInformation::PairwiseSimilarityMatrix(&psm),
            ),
        },
        None => panic!("Unsupported loss method: code = {}", loss),
    };
    let p = SALSOParameters {
        n_items,
        max_size,
        max_scans,
        n_runs,
        prob_sequential_allocation,
        prob_singletons_initialization,
        merge_split_strategy,
    };
    let results = minimize_by_salso(pdi, loss_function, p, secs, nanos, parallel, &mut rng);
    let results_slice = slice::from_raw_parts_mut(results_labels_ptr, n_items);
    for (i, v) in results.clustering.iter().enumerate() {
        results_slice[i] = i32::try_from(*v + 1).unwrap();
    }
    *results_expected_loss_ptr = results.expected_loss;
    *results_n_scans_ptr = i32::try_from(results.n_scans).unwrap();
    *results_n_merges_ptr = i32::try_from(results.n_merges).unwrap();
    *results_n_splits_ptr = i32::try_from(results.n_splits).unwrap();
    *results_n_runs_ptr = i32::try_from(results.n_runs).unwrap();
    *results_max_size_ptr = i32::try_from(results.max_size).unwrap();
    *results_initialization_method_ptr =
        i32::try_from(results.initialization_method.to_code()).unwrap();
}

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__minimize_by_enumeration(
    n_items: i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_label_ptr: *mut i32,
) {
    let ni = usize::try_from(n_items).unwrap();
    let psm = SquareMatrixBorrower::from_ptr(psm_ptr, ni);
    let f = match LossFunction::from_code(loss) {
        Some(loss_function) => match loss_function {
            LossFunction::BinderDraws => panic!("No implementation for binder2."),
            LossFunction::BinderPSM => binder_single_kernel,
            LossFunction::OneMinusARI => panic!("No implementation for omARI."),
            LossFunction::OneMinusARIapprox => omariapprox_single,
            LossFunction::VI => panic!("No implementation for VI."),
            LossFunction::VIlb => vilb_single_kernel,
        },
        None => panic!("Unsupported loss method: code = {}", loss),
    };
    let minimizer = minimize_by_enumeration(f, &psm);
    let results_slice = slice::from_raw_parts_mut(results_label_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v + 1).unwrap();
    }
}
