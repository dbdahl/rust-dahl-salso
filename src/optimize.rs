extern crate num_cpus;
extern crate rand;

use crate::loss::{binder_single, vilb_expected_loss_constant, vilb_single_kernel};
use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Beta, Distribution};
use rand_isaac::IsaacRng;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::slice;
use std::sync::mpsc;

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a.is_nan() {
        Ordering::Greater
    } else if b.is_nan() {
        Ordering::Less
    } else if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn cmp_f64_with_enumeration(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    if a.1.is_nan() {
        Ordering::Greater
    } else if b.1.is_nan() {
        Ordering::Less
    } else if a.1 < b.1 {
        Ordering::Less
    } else if a.1 > b.1 {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

pub trait Computer {
    fn new_subset(&mut self, partition: &mut Partition);
    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: usize) -> f64;
    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: usize);
    fn remove(&mut self, partition: &mut Partition, i: usize) -> usize;
    fn expected_loss(&self) -> f64;
    fn expected_loss_unnormalized(&self) -> f64;
    fn final_loss_from_kernel(&self, kernel: f64) -> f64;
}

// Expectation of the Binder Loss

#[derive(Debug)]
struct BinderSubsetCalculations {
    committed_loss: f64,
    speculative_loss: f64,
}

pub struct BinderComputer<'a> {
    subsets: Vec<BinderSubsetCalculations>,
    psm: &'a SquareMatrixBorrower<'a>,
}

impl<'a> BinderComputer<'a> {
    pub fn new(psm: &'a SquareMatrixBorrower<'a>) -> BinderComputer<'a> {
        BinderComputer {
            subsets: Vec::new(),
            psm,
        }
    }
}

impl<'a> Computer for BinderComputer<'a> {
    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(BinderSubsetCalculations {
            committed_loss: 0.0,
            speculative_loss: 0.0,
        });
    }

    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: usize) -> f64 {
        self.subsets[subset_index].speculative_loss = partition.subsets()[subset_index]
            .items()
            .iter()
            .fold(0.0, |s, j| {
                s + 0.5 - unsafe { *self.psm.get_unchecked((i, *j)) }
            });
        self.subsets[subset_index].speculative_loss
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: usize) {
        self.subsets[subset_index].committed_loss += self.subsets[subset_index].speculative_loss;
        partition.add_with_index(i, subset_index);
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> usize {
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
        subset_index
    }

    fn expected_loss(&self) -> f64 {
        2.0 * self.expected_loss_unnormalized() + self.psm.sum_of_triangle()
    }

    fn expected_loss_unnormalized(&self) -> f64 {
        self.subsets
            .iter()
            .fold(0.0, |s, subset| s + subset.committed_loss)
    }

    fn final_loss_from_kernel(&self, kernel: f64) -> f64 {
        2.0 * kernel + self.psm.sum_of_triangle()
    }
}

// Lower bound of the expectation of variation of information loss

#[derive(Debug)]
struct VarOfInforLBCacheUnit {
    item: usize,
    committed_sum: f64,
    committed_contribution: f64,
    speculative_sum: f64,
    speculative_contribution: f64,
}

#[derive(Debug)]
struct VarOfInfoLBSubsetCalculations {
    cached_units: Vec<VarOfInforLBCacheUnit>,
    committed_loss: f64,
    speculative_loss: f64,
}

pub struct VarOfInfoLBComputer<'a> {
    subsets: Vec<VarOfInfoLBSubsetCalculations>,
    psm: &'a SquareMatrixBorrower<'a>,
}

impl<'a> VarOfInfoLBComputer<'a> {
    pub fn new(psm: &'a SquareMatrixBorrower<'a>) -> VarOfInfoLBComputer<'a> {
        VarOfInfoLBComputer {
            subsets: Vec::new(),
            psm,
        }
    }
}

impl<'a> Computer for VarOfInfoLBComputer<'a> {
    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(VarOfInfoLBSubsetCalculations {
            cached_units: Vec::new(),
            committed_loss: 0.0,
            speculative_loss: 0.0,
        })
    }

    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: usize) -> f64 {
        let subset_of_partition = &partition.subsets()[subset_index];
        if subset_of_partition.n_items() == 0 {
            self.subsets[subset_index]
                .cached_units
                .push(VarOfInforLBCacheUnit {
                    item: i,
                    committed_sum: 0.0,
                    committed_contribution: 0.0,
                    speculative_sum: 1.0,
                    speculative_contribution: 0.0,
                });
            return 0.0;
        }
        for cu in self.subsets[subset_index].cached_units.iter_mut() {
            cu.speculative_sum =
                cu.committed_sum + unsafe { *self.psm.get_unchecked((cu.item, i)) };
            cu.speculative_contribution = cu.speculative_sum.log2();
        }
        let sum = subset_of_partition
            .items()
            .iter()
            .fold(0.0, |s, j| s + unsafe { *self.psm.get_unchecked((i, *j)) })
            + 1.0; // Because self.psm[(i, i)] == 1;
        self.subsets[subset_index]
            .cached_units
            .push(VarOfInforLBCacheUnit {
                item: i,
                committed_sum: 0.0,
                committed_contribution: 0.0,
                speculative_sum: sum,
                speculative_contribution: sum.log2(),
            });
        let nif = subset_of_partition.n_items() as f64;
        let s1 = (nif + 1.0) * (nif + 1.0).log2();
        let s2 = self.subsets[subset_index]
            .cached_units
            .iter()
            .fold(0.0, |s, cu| s + cu.speculative_contribution);
        self.subsets[subset_index].speculative_loss = s1 - 2.0 * s2;
        self.subsets[subset_index].speculative_loss - self.subsets[subset_index].committed_loss
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: usize) {
        for (index, subset) in self.subsets.iter_mut().enumerate() {
            if index == subset_index {
                for cu in subset.cached_units.iter_mut() {
                    cu.committed_sum = cu.speculative_sum;
                    cu.committed_contribution = cu.speculative_contribution;
                }
            } else {
                subset.cached_units.pop();
            }
        }
        self.subsets[subset_index].committed_loss = self.subsets[subset_index].speculative_loss;
        partition.add_with_index(i, subset_index);
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> usize {
        let subset_index = partition.label_of(i).unwrap();
        for cu in self.subsets[subset_index].cached_units.iter_mut() {
            cu.committed_sum -= unsafe { self.psm.get_unchecked((cu.item, i)) };
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
        subset_index
    }

    fn expected_loss(&self) -> f64 {
        let nif = self.psm.n_items() as f64;
        (self.expected_loss_unnormalized() + vilb_expected_loss_constant(self.psm)) / nif
    }

    fn expected_loss_unnormalized(&self) -> f64 {
        self.subsets
            .iter()
            .fold(0.0, |s, subset| s + subset.committed_loss)
    }

    fn final_loss_from_kernel(&self, kernel: f64) -> f64 {
        (kernel + vilb_expected_loss_constant(self.psm)) / (self.psm.n_items() as f64)
    }
}

// General algorithm

fn ensure_empty_subset<T: Computer>(partition: &mut Partition, computer: &mut T, max_label: usize) {
    match partition.subsets().last() {
        None => computer.new_subset(partition),
        Some(last) => {
            if !last.is_empty() && partition.n_subsets() <= max_label {
                computer.new_subset(partition)
            }
        }
    }
}

fn micro_optimized_allocation<T: Rng, U: Computer>(
    partition: &mut Partition,
    computer: &mut U,
    i: usize,
    probability_of_exploration: f64,
    rng: &mut T,
) -> usize {
    let max_label = partition.n_subsets() - 1;
    let mut iter = (0..=max_label)
        .map(|subset_index| computer.speculative_add(partition, i, subset_index))
        .enumerate();
    let take_best = if probability_of_exploration > 0.0 {
        rng.gen_range(0.0, 1.0) >= probability_of_exploration
    } else {
        true
    };
    let subset_index = if take_best {
        iter.min_by(cmp_f64_with_enumeration).unwrap().0
    } else {
        let mut first_best = iter.next().unwrap();
        let second_best_option = iter.next();
        if second_best_option.is_none() {
            first_best.0
        } else {
            let mut second_best = second_best_option.unwrap();
            if cmp_f64_with_enumeration(&first_best, &second_best) == Ordering::Greater {
                std::mem::swap(&mut first_best, &mut second_best);
            }
            for tuple in iter {
                if cmp_f64_with_enumeration(&tuple, &second_best) == Ordering::Less {
                    if cmp_f64_with_enumeration(&tuple, &first_best) == Ordering::Less {
                        second_best = first_best;
                        first_best = tuple;
                    } else {
                        second_best = tuple;
                    }
                }
            }
            second_best.0
        }
    };
    computer.add_with_index(partition, i, subset_index);
    subset_index
}

pub fn minimize_once_by_salso<'a, T: Rng, U: Computer>(
    computer_factory: Box<dyn Fn(&'a SquareMatrixBorrower<'a>) -> U>,
    max_label: usize,
    psm: &'a SquareMatrixBorrower<'a>,
    max_scans: u32,
    n_permutations: u32,
    probability_of_exploration: f64,
    stop_time: std::time::SystemTime,
    rng: &mut T,
) -> (Vec<usize>, f64, u32, u32) {
    let ni = psm.n_items();
    let beta_distribution_option = if probability_of_exploration > 0.0 {
        Some(Beta::new(1.0, 1.0 / probability_of_exploration).unwrap())
    } else {
        None
    };
    let mut global_minimum = std::f64::INFINITY;
    let mut global_best = Partition::new(ni);
    let mut global_n_scans = 0;
    let mut permutation: Vec<usize> = (0..ni).collect();
    let mut permutations_counter = 0;
    while permutations_counter < n_permutations {
        let mut computer = computer_factory(psm);
        let mut partition = Partition::new(ni);
        permutation.shuffle(rng);
        // Initial allocation
        let pr_explore = if max_scans == 0 {
            0.0
        } else {
            match beta_distribution_option {
                Some(beta) => beta.sample(rng),
                None => -probability_of_exploration,
            }
        };
        for i in 0..ni {
            ensure_empty_subset(&mut partition, &mut computer, max_label);
            let ii = unsafe { *permutation.get_unchecked(i) };
            micro_optimized_allocation(&mut partition, &mut computer, ii, pr_explore, rng);
        }
        // Sweetening scans
        let mut stop_exploring = false;
        let mut n_scans = 0;
        while n_scans < max_scans {
            if n_scans == max_scans - 1 {
                stop_exploring = true;
            }
            permutation.shuffle(rng);
            let mut no_change = true;
            let pr_explore = if stop_exploring {
                0.0
            } else {
                match beta_distribution_option {
                    Some(beta) => beta.sample(rng),
                    None => -probability_of_exploration,
                }
            };
            for i in 0..ni {
                ensure_empty_subset(&mut partition, &mut computer, max_label);
                let ii = unsafe { *permutation.get_unchecked(i) };
                let previous_subset_index = computer.remove(&mut partition, ii);
                let subset_index =
                    micro_optimized_allocation(&mut partition, &mut computer, ii, pr_explore, rng);
                if subset_index != previous_subset_index {
                    no_change = false;
                };
            }
            n_scans += 1;
            if no_change {
                if stop_exploring {
                    break;
                } else {
                    stop_exploring = true;
                }
            }
        }
        let value = computer.expected_loss_unnormalized();
        if value < global_minimum {
            global_minimum = value;
            global_best = partition;
            global_n_scans = n_scans;
        }
        permutations_counter += 1;
        if std::time::SystemTime::now() > stop_time {
            break;
        }
    }
    // Canonicalize the labels
    global_best.canonicalize();
    let labels = global_best.labels_via_copying();
    let loss = computer_factory(psm).final_loss_from_kernel(global_minimum);
    (labels, loss, global_n_scans, permutations_counter)
}

pub fn minimize_by_salso<'a, T: Rng>(
    psm: &'a SquareMatrixBorrower,
    use_vilb: bool,
    max_size: usize,
    max_scans: u32,
    n_permutations: u32,
    probability_of_exploration: f64,
    seconds: u64,
    nanoseconds: u32,
    parallel: bool,
    mut rng: &mut T,
) -> (Vec<usize>, f64, u32, u32) {
    let max_label = if max_size == 0 {
        usize::max_value()
    } else {
        max_size - 1
    };
    let stop_time = std::time::SystemTime::now() + std::time::Duration::new(seconds, nanoseconds);
    if !parallel {
        if use_vilb {
            minimize_once_by_salso(
                Box::new(|psm: &'a SquareMatrixBorrower<'a>| VarOfInfoLBComputer::new(psm)),
                max_label,
                psm,
                max_scans,
                n_permutations,
                probability_of_exploration,
                stop_time,
                rng,
            )
        } else {
            minimize_once_by_salso(
                Box::new(|psm: &'a SquareMatrixBorrower<'a>| BinderComputer::new(psm)),
                max_label,
                psm,
                max_scans,
                n_permutations,
                probability_of_exploration,
                stop_time,
                rng,
            )
        }
    } else {
        let (tx, rx) = mpsc::channel();
        let n_cores = num_cpus::get() as u32;
        let n_permutations = (n_permutations + n_cores - 1) / n_cores;
        crossbeam::scope(|s| {
            for _ in 0..n_cores {
                let tx = mpsc::Sender::clone(&tx);
                let mut child_rng = IsaacRng::from_rng(&mut rng).unwrap();
                s.spawn(move |_| {
                    let result = if use_vilb {
                        minimize_once_by_salso(
                            Box::new(|psm: &'a SquareMatrixBorrower<'a>| {
                                VarOfInfoLBComputer::new(psm)
                            }),
                            max_label,
                            psm,
                            max_scans,
                            n_permutations,
                            probability_of_exploration,
                            stop_time,
                            &mut child_rng,
                        )
                    } else {
                        minimize_once_by_salso(
                            Box::new(|psm: &'a SquareMatrixBorrower<'a>| BinderComputer::new(psm)),
                            max_label,
                            psm,
                            max_scans,
                            n_permutations,
                            probability_of_exploration,
                            stop_time,
                            &mut child_rng,
                        )
                    };
                    tx.send(result).unwrap();
                });
            }
        })
        .unwrap();
        std::mem::drop(tx); // Because of the cloning in the loop.
        let mut working_best = (vec![0usize; psm.n_items()], std::f64::INFINITY, 0, 0);
        let mut permutations_counter = 0;
        for candidate in rx {
            permutations_counter += candidate.3;
            if candidate.1 < working_best.1 {
                working_best = candidate;
            }
        }
        working_best.3 = permutations_counter;
        working_best
    }
}

pub fn minimize_by_enumeration(
    f: fn(&[usize], &SquareMatrixBorrower) -> f64,
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
                    let value = f(&partition[..], psm);
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
        let value = f(&partition[..], psm);
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
        let mut psm = SquareMatrix::identity(5);
        let psm_view = &psm.view();
        minimize_by_salso(
            psm_view,
            true,
            2,
            10,
            100,
            0.05,
            5,
            0,
            false,
            &mut thread_rng(),
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__minimize_by_salso(
    n_items: i32,
    psm_ptr: *mut f64,
    loss: i32,
    max_size: i32,
    max_scans: i32,
    n_permutations: i32,
    probability_of_exploration: f64,
    seconds: f64,
    parallel: i32,
    results_labels_ptr: *mut i32,
    results_expected_loss_ptr: *mut f64,
    results_scans_ptr: *mut i32,
    results_actual_n_permutations_ptr: *mut i32,
    seed_ptr: *const i32, // Assumed length is 32
) {
    let ni = usize::try_from(n_items).unwrap();
    let psm = SquareMatrixBorrower::from_ptr(psm_ptr, ni);
    let max_size = usize::try_from(max_size).unwrap();
    let max_scans = u32::try_from(max_scans).unwrap();
    let n_permutations = u32::try_from(n_permutations).unwrap();
    let (secs, nanos) = if seconds <= 0.0 {
        (1000 * 365 * 24 * 60 * 60, 0) // 1,000 years
    } else {
        (
            seconds.floor() as u64,
            ((seconds - seconds.floor()) * 1_000_000_000.0).floor() as u32,
        )
    };
    let parallel = parallel != 0;
    let mut rng = mk_rng_isaac(seed_ptr);
    let (minimizer, expected_loss, scans, actual_n_permutations) = minimize_by_salso(
        &psm,
        loss != 0,
        max_size,
        max_scans,
        n_permutations,
        probability_of_exploration,
        secs,
        nanos,
        parallel,
        &mut rng,
    );
    let results_slice = slice::from_raw_parts_mut(results_labels_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
    *results_expected_loss_ptr = expected_loss;
    *results_scans_ptr = i32::try_from(scans).unwrap();
    *results_actual_n_permutations_ptr = i32::try_from(actual_n_permutations).unwrap();
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
    let f = match loss {
        0 => binder_single,
        1 => vilb_single_kernel,
        _ => panic!("Unsupported loss method: {}", loss),
    };
    let minimizer = minimize_by_enumeration(f, &psm);
    let results_slice = slice::from_raw_parts_mut(results_label_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
}
